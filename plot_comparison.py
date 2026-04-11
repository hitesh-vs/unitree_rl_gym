import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

# --- Marker list for Robots ---
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]

def load_and_preprocess(prefix):
    X = np.load(f"{prefix}_embeds.npy")
    raw_labels = pickle.load(open(f"{prefix}_labels.pkl", "rb"))
    
    if len(X) > 5000:
        idx = np.random.RandomState(42).choice(len(X), 5000, replace=False)
        X, raw_labels = X[idx], [raw_labels[i] for i in idx]
    
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Z = TSNE(n_components=2, perplexity=40, metric="cosine", 
             n_iter=2000, random_state=42).fit_transform(X)
    
    # We now keep the full semantic string (e.g., 'left_ankle_pitch')
    semantics = [l["semantic"] for l in raw_labels]
    robots = [l["robot"] for l in raw_labels]
    
    return Z, semantics, robots

def scatter_panel(ax, Z, semantic_labels, robot_labels, title):
    unique_links = sorted(list(set(semantic_labels)))
    unique_robots = sorted(list(set(robot_labels)))
    
    # Create a large color palette for individual links
    cmap = plt.get_cmap("tab20") # tab20 has 20 distinct colors; use 'gist_rainbow' if you have more
    link2color = {link: cmap(i / len(unique_links)) for i, link in enumerate(unique_links)}
    robot2marker = {r: MARKERS[i % len(MARKERS)] for i, r in enumerate(unique_robots)}

    sem_arr = np.array(semantic_labels)
    rob_arr = np.array(robot_labels)

    for link in unique_links:
        for robot in unique_robots:
            mask = (sem_arr == link) & (rob_arr == robot)
            if mask.sum() == 0: continue
            ax.scatter(Z[mask, 0], Z[mask, 1], 
                       color=link2color[link],
                       marker=robot2marker[robot], 
                       alpha=0.7, s=25, linewidths=0)

    # LEGEND 1: Individual Links (Colors)
    # Note: If you have >20 links, this legend will be very tall. 
    # We use a smaller font and 2 columns.
    c_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=link2color[l], 
                 label=l, markersize=6) for l in unique_links]
    leg1 = ax.legend(handles=c_handles, title="Individual Links", 
                     loc="upper left", bbox_to_anchor=(1, 1), fontsize=6, ncol=1)
    ax.add_artist(leg1)

    # LEGEND 2: Robot Identity (Shapes)
    s_handles = [Line2D([0], [0], marker=robot2marker[r], color="#555555", linestyle='None',
                 label=r, markersize=7) for r in unique_robots]
    ax.legend(handles=s_handles, title="Robot", loc="lower left", fontsize=7)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--none_prefix", required=True)
    parser.add_argument("--rwse_prefix", required=True)
    parser.add_argument("--out", default="comparison_individual_links.png")
    args = parser.parse_args()

    Z_n, S_n, R_n = load_and_preprocess(args.none_prefix)
    Z_r, S_r, R_r = load_and_preprocess(args.rwse_prefix)

    # Increased width (figsize) to accommodate the long legend on the right
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    scatter_panel(axes[0], Z_n, S_n, R_n, "Identity (No Encoding)")
    scatter_panel(axes[1], Z_r, S_r, R_r, "RWSE (Structural Encoding)")

    plt.suptitle("Fine-grained Link Embeddings: Identity vs RWSE", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Detailed link plot saved to: {args.out}")

if __name__ == "__main__":
    main()