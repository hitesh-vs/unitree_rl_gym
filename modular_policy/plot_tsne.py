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

# Fixed palette for the 4 coarse joint roles — colourblind-friendly
COARSE_COLORS = {
    "root":  "#222222",   # near-black
    "hip":   "#2166ac",   # blue
    "knee":  "#1a9850",   # green
    "ankle": "#d6604d",   # orange-red
    "other": "#aaaaaa",   # light grey (filtered out by default)
}
COARSE_ORDER = ["root", "hip", "knee", "ankle", "other"]


def coarsen_semantic(sem):
    """Collapse detailed left_hip_pitch / right_knee / … into 4 coarse roles."""
    s = sem.lower()
    if "root" in s or "pelvis" in s or "torso" in s:
        return "root"
    if "ankle" in s:
        return "ankle"
    if "knee" in s:
        return "knee"
    if "hip" in s:
        return "hip"
    return "other"


def load_data(prefix):
    embeds = np.load(f"{prefix}_embeds.npy")
    labels = pickle.load(open(f"{prefix}_labels.pkl", "rb"))
    return embeds, labels


def run_tsne(X, labels, perplexity=40, n_iter=2000, max_samples=5000):
    """
    Subsample, normalise, run t-SNE.
    Returns (Z, subsampled_labels) so callers stay in sync.
    """
    if len(X) > max_samples:
        idx    = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X      = X[idx]
        labels = [labels[i] for i in idx]
        print(f"  Subsampled to {max_samples} points")

    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"  Running t-SNE on {X.shape[0]} × {X.shape[1]} ...", flush=True)

    try:
        Z = TSNE(
            n_components=2, perplexity=perplexity,
            metric="cosine", n_iter=n_iter, random_state=42,
        ).fit_transform(X)
    except TypeError:
        Z = TSNE(
            n_components=2, perplexity=perplexity,
            metric="cosine", max_iter=n_iter, random_state=42,
        ).fit_transform(X)

    return Z, labels


def filter_coarse(Z, labels):
    """
    Build coarse semantic list, drop 'other', return filtered
    (Z, labels, coarse_semantic, robots).
    """
    coarse = [coarsen_semantic(l["semantic"]) for l in labels]
    robots = [l["robot"] for l in labels]

    keep   = [i for i, c in enumerate(coarse) if c != "other"]
    Z      = Z[keep]
    labels = [labels[i] for i in keep]
    coarse = [coarse[i] for i in keep]
    robots = [robots[i] for i in keep]
    return Z, labels, coarse, robots


def knn_purity(Z, label_arr, k=16):
    Z         = np.array(Z, dtype=np.float32)
    label_arr = np.array(label_arr)
    nn        = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Z)
    _, idx    = nn.kneighbors(Z)
    idx       = idx[:, 1:]
    return float(np.mean([
        np.mean(label_arr[idx[i]] == label_arr[i])
        for i in range(len(label_arr))
    ]))


def scatter_panel(ax, Z, coarse_labels, shape_labels, title):
    """
    Color = coarse joint role (fixed palette)
    Shape = robot identity
    """
    unique_roles  = [r for r in COARSE_ORDER
                     if r in set(coarse_labels) and r != "other"]
    unique_robots = sorted(set(shape_labels))

    robot2marker = {r: MARKERS[i % len(MARKERS)]
                    for i, r in enumerate(unique_robots)}

    coarse_arr = np.array(coarse_labels)
    shape_arr  = np.array(shape_labels)

    # Plot in role order so legend is consistent
    for role in unique_roles:
        for robot in unique_robots:
            mask = (coarse_arr == role) & (shape_arr == robot)
            if mask.sum() == 0:
                continue
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=COARSE_COLORS[role],
                       marker=robot2marker[robot],
                       alpha=0.65, s=40, linewidths=0,
                       label="_nolegend_")

    # Color legend (joint role)
    color_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COARSE_COLORS[r], markersize=10,
               label=r, markeredgewidth=0)
        for r in unique_roles
    ]
    # Shape legend (robot)
    shape_handles = [
        Line2D([0], [0], marker=robot2marker[r], color="#555555",
               markersize=9, label=r, linestyle="None")
        for r in unique_robots
    ]

    leg1 = ax.legend(handles=color_handles, title="Joint role",
                     fontsize=7, loc="upper left", framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title="Robot",
              fontsize=7, loc="upper right", framealpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def print_purity(Z, coarse, robots, tag=""):
    print(f"\n{'─'*50}")
    if tag:
        print(f"  {tag}")
    for arr, name in [(coarse, "joint role"), (robots, "robot")]:
        a      = np.array(arr)
        p      = knn_purity(Z, a)
        chance = 1.0 / len(set(arr))
        print(f"  KNN-16 purity [{name:>12s}]: {p:.3f}  (chance={chance:.3f})")


def plot_single(prefix, out_path, title_tag="", perplexity=40):
    embeds, labels            = load_data(prefix)
    Z, labels                 = run_tsne(embeds, labels, perplexity=perplexity)
    Z, labels, coarse, robots = filter_coarse(Z, labels)

    print_purity(Z, coarse, robots, tag=title_tag or os.path.basename(prefix))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter_panel(axes[0], Z, coarse, robots,
                  "Joint role (color) × Robot (shape)\n← want: same role clusters together")
    scatter_panel(axes[1], Z, robots, robots,
                  "Robot identity\n← want: mixed across clusters")

    sup = title_tag or os.path.basename(prefix)
    plt.suptitle(f"t-SNE of transformer input embedding — {sup}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {out_path}")
    plt.close()


def plot_comparison(prefix_a, prefix_b, label_a, label_b, out_path, perplexity=40):
    """Side-by-side comparison of two runs (e.g. iter 10 vs iter 250)."""
    embeds_a, labels_a = load_data(prefix_a)
    embeds_b, labels_b = load_data(prefix_b)

    print(f"Run A ({label_a}): {len(embeds_a)} embeddings")
    print(f"Run B ({label_b}): {len(embeds_b)} embeddings")

    Za, labels_a = run_tsne(embeds_a, labels_a, perplexity=perplexity)
    Zb, labels_b = run_tsne(embeds_b, labels_b, perplexity=perplexity)

    Za, labels_a, coarse_a, robots_a = filter_coarse(Za, labels_a)
    Zb, labels_b, coarse_b, robots_b = filter_coarse(Zb, labels_b)

    print_purity(Za, coarse_a, robots_a, tag=label_a)
    print_purity(Zb, coarse_b, robots_b, tag=label_b)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    scatter_panel(axes[0, 0], Za, coarse_a, robots_a,
                  f"{label_a}\nJoint role × Robot")
    scatter_panel(axes[0, 1], Za, robots_a, robots_a,
                  f"{label_a}\nRobot identity")
    scatter_panel(axes[1, 0], Zb, coarse_b, robots_b,
                  f"{label_b}\nJoint role × Robot")
    scatter_panel(axes[1, 1], Zb, robots_b, robots_b,
                  f"{label_b}\nRobot identity")

    plt.suptitle(
        f"t-SNE comparison: {label_a} vs {label_b}\n"
        f"Key metric: joint role purity (higher = similar joints cluster together)",
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
    parser.add_argument("--labels",         nargs=2, default=["Run A", "Run B"],
                        help="Display names for the two runs")
    parser.add_argument("--out",            default=None,
                        help="Output PNG path (default: prefix_tsne.png)")
    parser.add_argument("--perplexity",     type=int, default=40)
    parser.add_argument("--max_samples",    type=int, default=5000,
                        help="Max points for t-SNE (default: 5000)")
    args = parser.parse_args()

    import functools
    global run_tsne
    _orig    = run_tsne
    run_tsne = functools.partial(_orig, max_samples=args.max_samples)

    out_path = args.out or f"{args.prefix}_tsne.png"

    if args.compare_prefix:
        plot_comparison(
            args.prefix, args.compare_prefix,
            args.labels[0], args.labels[1],
            out_path, perplexity=args.perplexity)
    else:
        plot_single(args.prefix, out_path,
                    title_tag=args.labels[0],
                    perplexity=args.perplexity)


if __name__ == "__main__":
    main()