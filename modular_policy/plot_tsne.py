"""
modular_policy/plot_tsne.py

Standalone t-SNE plot from saved embeddings.
Supports comparing multiple checkpoints (e.g. topological vs rwse).

Usage:
    # Single run
    python modular_policy/plot_tsne.py \
        --prefix output_walk_isaac/Mar21_18-51-07/tsne_iter0400

    # Compare two runs side by side
    python modular_policy/plot_tsne.py \
        --prefix output_topological/tsne_iter0400 \
        --compare_prefix output_rwse/tsne_iter0400 \
        --labels "Topological" "RWSE"
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]


def load_data(prefix):
    embeds = np.load(f"{prefix}_embeds.npy")
    labels = pickle.load(open(f"{prefix}_labels.pkl", "rb"))
    return embeds, labels


def run_tsne(X, perplexity=40, n_iter=2000):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"  Running t-SNE on {X.shape[0]} × {X.shape[1]} ...", flush=True)
    return TSNE(
        n_components=2, perplexity=perplexity,
        metric="cosine", max_iter=n_iter, random_state=42,
    ).fit_transform(X)


def knn_purity(Z, label_arr, k=16):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Z)
    _, idx = nn.kneighbors(Z)
    idx    = idx[:, 1:]
    return float(np.mean([
        np.mean(label_arr[idx[i]] == label_arr[i])
        for i in range(len(label_arr))
    ]))


def scatter_panel(ax, Z, color_labels, shape_labels, title, max_classes=20):
    """
    Color = semantic joint type
    Shape = robot identity
    This is the key plot: if RWSE works, same joint types from different
    robots should cluster together (mixed shapes, same color).
    """
    unique_colors = sorted(set(color_labels))
    unique_shapes = sorted(set(shape_labels))

    cmap        = cm.get_cmap("tab20", min(len(unique_colors), max_classes))
    label2color = {l: cmap(i) for i, l in enumerate(unique_colors)}
    robot2marker = {r: MARKERS[i % len(MARKERS)]
                    for i, r in enumerate(unique_shapes)}

    for robot in unique_shapes:
        for lbl in unique_colors:
            mask = (color_labels == lbl) & (shape_labels == robot)
            if mask.sum() == 0:
                continue
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=[label2color[lbl]],
                       marker=robot2marker[robot],
                       alpha=0.6, s=35, linewidths=0)

    # Color legend (joint type)
    color_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=label2color[l], markersize=8, label=l)
        for l in unique_colors
    ]
    # Shape legend (robot)
    shape_handles = [
        Line2D([0], [0], marker=robot2marker[r], color="gray",
               markersize=8, label=r)
        for r in unique_shapes
    ]
    leg1 = ax.legend(handles=color_handles, title="Joint type",
                     fontsize=6, loc="upper left",
                     framealpha=0.7)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title="Robot",
              fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def print_purity(Z, labels, tag=""):
    semantic = np.array([l["semantic"] for l in labels])
    robots   = np.array([l["robot"]   for l in labels])

    print(f"\n{'─'*50}")
    if tag:
        print(f"  {tag}")
    for arr, name in [(semantic, "semantic"), (robots, "robot")]:
        p     = knn_purity(Z, arr)
        chance = 1.0 / len(set(arr))
        print(f"  KNN-16 purity [{name:>10s}]: {p:.3f}  (chance={chance:.3f})")


def plot_single(prefix, out_path, title_tag=""):
    embeds, labels = load_data(prefix)

    semantic = np.array([l["semantic"] for l in labels])
    robots   = np.array([l["robot"]   for l in labels])

    Z = run_tsne(embeds)
    print_purity(Z, labels, tag=title_tag or os.path.basename(prefix))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter_panel(axes[0], Z, semantic, robots,
                  "Joint type  (color) × Robot (shape)\n← want: same color clusters together")
    scatter_panel(axes[1], Z, robots, robots,
                  "Robot identity  ← want: MIXED across clusters")

    sup = title_tag or os.path.basename(prefix)
    plt.suptitle(f"t-SNE of transformer input embedding — {sup}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {out_path}")
    plt.close()


def plot_comparison(prefix_a, prefix_b, label_a, label_b, out_path):
    """Side-by-side comparison of two runs (e.g. topological vs rwse)."""
    embeds_a, labels_a = load_data(prefix_a)
    embeds_b, labels_b = load_data(prefix_b)

    print(f"Run A ({label_a}): {len(embeds_a)} embeddings")
    print(f"Run B ({label_b}): {len(embeds_b)} embeddings")

    Za = run_tsne(embeds_a)
    Zb = run_tsne(embeds_b)

    print_purity(Za, labels_a, tag=label_a)
    print_purity(Zb, labels_b, tag=label_b)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    sem_a = np.array([l["semantic"] for l in labels_a])
    rob_a = np.array([l["robot"]   for l in labels_a])
    sem_b = np.array([l["semantic"] for l in labels_b])
    rob_b = np.array([l["robot"]   for l in labels_b])

    scatter_panel(axes[0, 0], Za, sem_a, rob_a,
                  f"{label_a}\nJoint type × Robot")
    scatter_panel(axes[0, 1], Za, rob_a, rob_a,
                  f"{label_a}\nRobot identity")
    scatter_panel(axes[1, 0], Zb, sem_b, rob_b,
                  f"{label_b}\nJoint type × Robot")
    scatter_panel(axes[1, 1], Zb, rob_b, rob_b,
                  f"{label_b}\nRobot identity")

    plt.suptitle(
        f"t-SNE comparison: {label_a} vs {label_b}\n"
        f"Key metric: joint type purity (higher = similar joints cluster together)",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",         required=True,
                        help="Path prefix for *_embeds.npy and *_labels.pkl")
    parser.add_argument("--compare_prefix", default=None,
                        help="Second run to compare against (optional)")
    parser.add_argument("--labels",         nargs=2,
                        default=["Run A", "Run B"],
                        help="Display names for the two runs")
    parser.add_argument("--out",            default=None,
                        help="Output PNG path (default: prefix_tsne.png)")
    parser.add_argument("--perplexity",     type=int, default=40)
    args = parser.parse_args()

    out_path = args.out or f"{args.prefix}_tsne.png"

    if args.compare_prefix:
        plot_comparison(
            args.prefix, args.compare_prefix,
            args.labels[0], args.labels[1],
            out_path)
    else:
        plot_single(args.prefix, out_path, title_tag=args.labels[0])


if __name__ == "__main__":
    main()