"""
plot_difficulty_vs_performance.py

Plots episode length vs morphology difficulty score for baseline vs FiLM.
Shows the trend across the full variant distribution.

Usage:
    python plot_difficulty_vs_performance.py \
        --changes_dir resources/robots/g1_variants_wide \
        --output difficulty_vs_performance.pdf
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Hardcode your results here ────────────────────────────────────────────────
# From your training logs — ep_len per variant
# Fill in both baseline and FiLM numbers

RESULTS = {
    # variant_name: (difficulty, baseline_ep_len, film_ep_len)
    # difficulty computed by variant_difficulty()
    # ep_len from training logs (avg ep len column)
    "robot_variant_0": {"baseline": 1001, "film": 998},
    "robot_variant_1": {"baseline": 1001, "film": 1001},
    "robot_variant_2": {"baseline":  938, "film": 1001},
    "robot_variant_3": {"baseline": 1001, "film":  916},
    "robot_variant_4": {"baseline":  454, "film":  882},
    "robot_variant_5": {"baseline": 1001, "film":  942},
    "robot_variant_6": {"baseline":  812, "film":  979},
    "robot_variant_7": {"baseline":  955, "film": 1001},
    "robot_variant_8": {"baseline": 1001, "film": 1001},
    "robot_variant_9": {"baseline": 1001, "film":  958},
    "g1_12dof":        {"baseline":  933, "film": 1001},
}


def variant_difficulty(changes_path):
    """
    Compute a single morphology difficulty score from changes.json.
    Higher = more different from base robot.
    """
    with open(changes_path) as f:
        c = json.load(f)

    scores = []

    # Mass: deviation from 1.0 + left/right asymmetry
    lm = c["group_mass_scale"]["left_leg"]
    rm = c["group_mass_scale"]["right_leg"]
    tm = c["group_mass_scale"]["torso"]
    scores.append(abs((lm + rm) / 2 - 1))       # average deviation
    scores.append(abs(lm - rm) / max(lm, rm))    # asymmetry ratio

    # Length: deviation from 1.0
    ll = c["group_length_scale"]["left_leg"]
    rl = c["group_length_scale"]["right_leg"]
    scores.append(abs((ll + rl) / 2 - 1))

    # Joint range: mean deviation
    jr = list(c["joint_range_scale"].values())
    scores.append(abs(np.mean(jr) - 1))
    scores.append(np.std(jr))                    # variance across joints

    # Effort: mean deviation
    ef = list(c["joint_effort_scale"].values())
    scores.append(abs(np.mean(ef) - 1))
    scores.append(np.std(ef))

    # Damping: mean deviation from base (0.001)
    damp = list(c["joint_damping"].values())
    base_damp = 0.001
    scores.append(np.mean([abs(d - base_damp) / base_damp for d in damp]))

    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--changes_dir", required=True,
                        help="Dir containing robot_variant_X_changes.json files")
    parser.add_argument("--output", default="difficulty_vs_performance.pdf")
    args = parser.parse_args()

    # ── Compute difficulty scores ─────────────────────────────────────────
    difficulties = {}
    for name in RESULTS:
        if name == "g1_12dof":
            difficulties[name] = 0.0   # base robot = zero difficulty
            continue
        changes_path = os.path.join(
            args.changes_dir, f"{name}_changes.json")
        if os.path.exists(changes_path):
            difficulties[name] = variant_difficulty(changes_path)
        else:
            print(f"Warning: {changes_path} not found, skipping")

    # ── Assemble plot data ────────────────────────────────────────────────
    xs      = []
    ys_base = []
    ys_film = []
    names   = []

    for name, res in RESULTS.items():
        if name not in difficulties:
            continue
        xs.append(difficulties[name])
        ys_base.append(res["baseline"])
        ys_film.append(res["film"])
        names.append(name)

    xs      = np.array(xs)
    ys_base = np.array(ys_base, dtype=float)
    ys_film = np.array(ys_film, dtype=float)

    # Sort by difficulty
    order   = np.argsort(xs)
    xs      = xs[order]
    ys_base = ys_base[order]
    ys_film = ys_film[order]
    names   = [names[i] for i in order]

    print("\nVariants sorted by difficulty:")
    print(f"{'Variant':<22} {'Difficulty':>10} {'Baseline':>10} {'FiLM':>8} {'Diff':>8}")
    print("-" * 62)
    for n, x, yb, yf in zip(names, xs, ys_base, ys_film):
        print(f"{n:<22} {x:>10.3f} {yb:>10.0f} {yf:>8.0f} {yf-yb:>+8.0f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter points
    ax.scatter(xs, ys_base, color="#E07B54", s=80, zorder=5,
               label="Baseline (Context + Transformer)")
    ax.scatter(xs, ys_film,  color="#4A90D9", s=80, zorder=5,
               label="Ours (Graph+RWSE+FiLM)")

    # Trend lines (polynomial fit degree 2)
    if len(xs) >= 3:
        x_smooth = np.linspace(xs.min(), xs.max(), 200)

        z_base = np.polyfit(xs, ys_base, 2)
        p_base = np.poly1d(z_base)
        ax.plot(x_smooth, p_base(x_smooth).clip(0, 1001),
                color="#E07B54", linewidth=2, linestyle="--", alpha=0.7)

        z_film = np.polyfit(xs, ys_film, 2)
        p_film = np.poly1d(z_film)
        ax.plot(x_smooth, p_film(x_smooth).clip(0, 1001),
                color="#4A90D9", linewidth=2, linestyle="-", alpha=0.7)

    # Shade difficulty regions
    x_min, x_max = xs.min() - 0.02, xs.max() + 0.02
    ax.axvspan(x_min,  0.18, alpha=0.06, color="green",  label="Easy")
    ax.axvspan(0.18,   0.35, alpha=0.06, color="orange", label="Medium")
    ax.axvspan(0.35,  x_max, alpha=0.06, color="red",    label="Hard")

    # Region labels
    ax.text(0.09,  50, "Easy",   fontsize=9, color="green",  alpha=0.8)
    ax.text(0.235, 50, "Medium", fontsize=9, color="orange", alpha=0.8)
    ax.text(0.39,  50, "Hard",   fontsize=9, color="red",    alpha=0.8)

    # Max ep line
    ax.axhline(1001, color="gray", linewidth=0.8,
               linestyle=":", alpha=0.5, label="Max episode length")

    # Labels
    ax.set_xlabel("Morphology Difficulty Score", fontsize=12)
    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.set_title("Generalization vs Morphology Difficulty", fontsize=13)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1100)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.output}")
    plt.show()


if __name__ == "__main__":
    main()