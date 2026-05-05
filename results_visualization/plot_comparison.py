import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

# --- Configuration ---
# Markers for different robots
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]

# Mapping fine-grained semantics to high-level groups
SEMANTIC_GROUPS = {
    "ankle": ["ankle"],
    "knee":  ["knee"],
    "hip":   ["hip"],
    "shoulder": ["shoulder"],
    "elbow": ["elbow"],
    "wrist": ["wrist"],
    "head":  ["head", "neck"],
    "foot":  ["foot", "toe"]
}

def get_high_level_group(semantic_str):
    s = semantic_str.lower()
    for group, keywords in SEMANTIC_GROUPS.items():
        if any(k in s for k in keywords):
            return group.capitalize()
    return "Other"

def load_and_preprocess(prefix):
    X = np.load(f"{prefix}_embeds.npy")
    raw_labels = pickle.load(open(f"{prefix}_labels.pkl", "rb"))

    # Filter out non-limbs
    keep = [i for i, l in enumerate(raw_labels)
            if l["semantic"] not in ("root", "center_root", "pad")
            and "pelvis" not in l["body"].lower()]
    X = X[keep]
    raw_labels = [raw_labels[i] for i in keep]
    
    if len(X) > 5000:
        idx = np.random.RandomState(42).choice(len(X), 5000, replace=False)
        X, raw_labels = X[idx], [raw_labels[i] for i in idx]
    
    # Cosine normalization + t-SNE
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Z = TSNE(n_components=2, perplexity=40, metric="cosine", 
             n_iter=2000, random_state=42).fit_transform(X)
    
    groups = [get_high_level_group(l["semantic"]) for l in raw_labels]
    robots = [l["robot"] for l in raw_labels]
    
    return Z, groups, robots

def scatter_panel(ax, Z, groups, robot_labels, title):
    unique_groups = sorted(list(set(groups)))
    unique_robots = sorted(list(set(robot_labels)))
    
    # Paper-friendly Muted Colors (Tableau 10 Light/Medium)
    # Colors: Muted blue, orange, green, red, purple, brown, pink, gray, olive
    muted_colors = [
        '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', 
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'
    ]
    group2color = {g: muted_colors[i % len(muted_colors)] for i, g in enumerate(unique_groups)}
    robot2marker = {r: MARKERS[i % len(MARKERS)] for i, r in enumerate(unique_robots)}

    grp_arr = np.array(groups)
    rob_arr = np.array(robot_labels)

    for g in unique_groups:
        for r in unique_robots:
            mask = (grp_arr == g) & (rob_arr == r)
            if mask.sum() == 0: continue
            ax.scatter(Z[mask, 0], Z[mask, 1], 
                       color=group2color[g],
                       marker=robot2marker[r], 
                       alpha=0.55, s=30, edgecolors='none')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Internal Panel Legend (Joint Groups)
    group_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, 
                     label=g, markersize=8) for g, c in group2color.items()]
    leg1 = ax.legend(handles=group_handles, title="Joint Group", loc="upper left", 
                     bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.add_artist(leg1)

    # Internal Panel Legend (Robot Identity)
    robot_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="#7f7f7f", 
                     markeredgecolor="#7f7f7f", label=r, markersize=7) for r, m in robot2marker.items()]
    ax.legend(handles=robot_handles, title="Robot", loc="lower left", 
              bbox_to_anchor=(1.01, 0), fontsize=8, frameon=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--none_prefix", required=True)
    parser.add_argument("--rwse_prefix", required=True)
    parser.add_argument("--out", default="tsne_comparison.pdf") # PDF is better for papers
    args = parser.parse_args()

    Z_n, G_n, R_n = load_and_preprocess(args.none_prefix)
    Z_r, G_r, R_r = load_and_preprocess(args.rwse_prefix)

    # Adjusted width to accommodate the side-by-side internal legends
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter_panel(axes[0], Z_n, G_n, R_n, "Identity (No Positional Encoding)")
    scatter_panel(axes[1], Z_r, G_r, R_r, "RWSE (Structural Positional Encoding)")

    plt.suptitle("Clustering of Functional Link Groups across Robot Morphologies", 
                 fontsize=14, y=0.98, fontweight='bold')
    
    # Using tight_layout with rect to make room for titles/suptext
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Paper-ready plot saved to: {args.out}")

if __name__ == "__main__":
    main()