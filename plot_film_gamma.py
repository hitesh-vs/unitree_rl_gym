"""
plot_film_gamma.py

Extracts FiLM gamma values per limb per variant and plots a heatmap.
Shows how FiLM modulates each limb's embedding differently per robot.

Usage:
    python plot_film_gamma.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --out film_gamma_heatmap.pdf \
        --num_envs 512
"""

import isaacgym
from isaacgym import gymapi

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


LIMB_NAMES = [
    "pelvis",
    "L_hip_p", "L_hip_r", "L_hip_y",
    "L_knee",
    "L_ank_p", "L_ank_r",
    "R_hip_p", "R_hip_r", "R_hip_y",
    "R_knee",
    "R_ank_p", "R_ank_r",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--xml_path",          required=True)
    p.add_argument("--variants_metadata", required=True)
    p.add_argument("--out",               default="film_gamma_heatmap.pdf")
    p.add_argument("--num_envs",          type=int, default=512)
    p.add_argument("--sim_device",        default="cuda:0")
    p.add_argument("--rl_device",         default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    from modular_policy.config import cfg
    from modular_policy.algos.ppo.runner import ModularRunner
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
    from legged_gym.utils.helpers import parse_sim_params, class_to_dict, set_seed

    with open(args.variants_metadata) as f:
        meta = json.load(f)
    variant_names = list(meta.keys())

    # ── cfg ───────────────────────────────────────────────────────────────
    cfg.PPO.NUM_ENVS               = args.num_envs
    cfg.MODEL.MAX_LIMBS            = 13
    cfg.MODEL.MAX_JOINTS           = 12
    cfg.MODEL.GRAPH_ENCODING       = "rwse"
    cfg.MODEL.RWSE_K               = 8
    cfg.MODEL.TRANSFORMER.USE_FILM = True
    cfg.MODEL.GCN.HIDDEN_DIM       = 16
    cfg.MODEL.GCN.OUT_DIM          = 13
    cfg.MODEL.GCN.NUM_LAYERS       = 4
    cfg.DEVICE                     = args.rl_device
    cfg.ENV.WALKERS                = variant_names

    # ── Create env ────────────────────────────────────────────────────────
    set_seed(1409)
    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs

    class _Args:
        physics_engine    = gymapi.SIM_PHYSX
        sim_device        = args.sim_device
        rl_device         = args.rl_device
        headless          = True
        use_gpu           = True
        use_gpu_pipeline  = True
        subscenes         = 0
        num_threads       = 10
        num_envs          = None
        seed              = None
        max_iterations    = None
        resume            = False
        experiment_name   = None
        run_name          = None
        load_run          = None
        checkpoint        = None
        device            = args.rl_device
        sim_device_type   = "cuda"
        compute_device_id = 0
        num_subscenes     = 0

    sim_params = parse_sim_params(_Args(), {"sim": class_to_dict(env_cfg.sim)})
    env = MultiVariantG1Robot(
        cfg                    = env_cfg,
        sim_params             = sim_params,
        physics_engine         = gymapi.SIM_PHYSX,
        sim_device             = args.sim_device,
        headless               = True,
        variants_metadata_path = args.variants_metadata,
    )
    print(f"Env ready — {env.num_envs} envs, {len(variant_names)} variants")

    # ── Runner + checkpoint ───────────────────────────────────────────────
    log_dir = f"/tmp/gamma_{datetime.now().strftime('%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    runner = ModularRunner(
        env                    = env,
        xml_path               = os.path.abspath(args.xml_path),
        log_dir                = log_dir,
        device                 = args.rl_device,
        variants_metadata_path = args.variants_metadata,
    )
    runner.load(args.checkpoint)
    print(f"Checkpoint loaded: {args.checkpoint}")

    # Check FiLM generator exists
    fg = runner.actor_critic.mu_net.film_generator
    if fg is None:
        print("ERROR: No FiLM generator found in checkpoint. "
              "Make sure this is a FiLM checkpoint.")
        return

    # ── Get obs ───────────────────────────────────────────────────────────
    env.reset_idx(torch.arange(env.num_envs, device=torch.device(args.rl_device)))
    runner.commands[:, 0] = 0.5
    obs = runner._get_obs_normalized()

    # ── Extract gamma per limb per variant ────────────────────────────────
    # obs["context"] shape: (N, max_limbs * 12)
    # Reshape to (max_limbs, N, 12) for FiLM generator
    N       = env.num_envs
    obs_ctx = obs["context"].reshape(N, cfg.MODEL.MAX_LIMBS, -1).permute(1, 0, 2)

    runner.actor_critic.eval()
    with torch.no_grad():
        gamma, beta = fg(obs_ctx)   # both (max_limbs, N, d_model)

    # Mean gamma over d_model → (max_limbs, N)
    gamma_per_limb = gamma.mean(dim=-1)   # (13, N)

    # Collect per-variant mean gamma
    variant_gammas  = []
    variant_labels  = []
    variant_gammas_std = []

    for v, vname in enumerate(variant_names):
        mask = (env.env_variant_ids == v)
        if mask.sum() == 0:
            continue
        g     = gamma_per_limb[:, mask]        # (13, n_envs_for_v)
        g_mean = g.mean(dim=1).cpu().numpy()   # (13,)
        g_std  = g.std(dim=1).cpu().numpy()    # (13,)
        variant_gammas.append(g_mean)
        variant_gammas_std.append(g_std)
        variant_labels.append(vname)
        print(f"  {vname}: gamma mean={g_mean.mean():.3f}  "
              f"std={g_mean.std():.3f}  "
              f"min={g_mean.min():.3f}  max={g_mean.max():.3f}")

    gamma_matrix = np.array(variant_gammas)      # (V, 13)
    std_matrix   = np.array(variant_gammas_std)  # (V, 13)
    n_variants   = len(variant_labels)

    print(f"\nGamma matrix shape: {gamma_matrix.shape}")
    print(f"Global gamma: mean={gamma_matrix.mean():.3f}  "
          f"min={gamma_matrix.min():.3f}  max={gamma_matrix.max():.3f}")

    # ── Plot heatmap ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_variants * 0.6 + 2)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left: gamma heatmap
    ax = axes[0]
    # Replace the norm lines with:
    g_min = gamma_matrix.min()
    g_max = gamma_matrix.max()
    vmin  = min(g_min - 0.02, 0.95)
    vmax  = max(g_max + 0.02, 1.05)
    vcenter = 1.0

    # Ensure strict ordering
    if vmin >= vcenter:
        vmin = vcenter - 0.05
    if vmax <= vcenter:
        vmax = vcenter + 0.05

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    im = ax.imshow(gamma_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="FiLM gamma\n(< 1 = suppress, > 1 = amplify)")

    ax.set_xticks(range(13))
    ax.set_xticklabels(LIMB_NAMES[:13], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_variants))
    ax.set_yticklabels(variant_labels, fontsize=9)
    ax.set_xlabel("Limb", fontsize=11)
    ax.set_ylabel("Variant", fontsize=11)
    ax.set_title("FiLM gamma per limb per variant\n"
                 "(diverging from 1.0 = active modulation)", fontsize=11)

    # Add value annotations
    for i in range(n_variants):
        for j in range(13):
            val = gamma_matrix[i, j]
            color = "white" if abs(val - 1.0) > 0.1 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Right: per-variant asymmetry score
    # Asymmetry = mean(|left_gamma - right_gamma|) across paired joints
    ax2 = axes[1]
    left_idx  = [1, 2, 3, 4, 5, 6]   # L_hip_p, L_hip_r, L_hip_y, L_knee, L_ank_p, L_ank_r
    right_idx = [7, 8, 9, 10, 11, 12] # R_hip_p, R_hip_r, R_hip_y, R_knee, R_ank_p, R_ank_r

    asymmetry = np.abs(
        gamma_matrix[:, left_idx] - gamma_matrix[:, right_idx]
    ).mean(axis=1)  # (V,)

    colors = plt.cm.Reds(asymmetry / max(asymmetry.max(), 0.01))
    bars = ax2.barh(range(n_variants), asymmetry, color=colors, edgecolor="none")
    ax2.set_yticks(range(n_variants))
    ax2.set_yticklabels(variant_labels, fontsize=9)
    ax2.set_xlabel("L/R asymmetry\n(mean |L_gamma - R_gamma|)", fontsize=9)
    ax2.set_title("FiLM asymmetry\nper variant", fontsize=11)
    ax2.axvline(0, color="gray", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, asymmetry)):
        ax2.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Variant':<22} {'Mean gamma':>10} {'Std gamma':>10} "
          f"{'L/R asym':>10} {'Max limb diff':>14}")
    print(f"{'-'*70}")
    for i, vname in enumerate(variant_labels):
        g      = gamma_matrix[i]
        l_mean = g[left_idx].mean()
        r_mean = g[right_idx].mean()
        asym   = asymmetry[i]
        maxdif = g.max() - g.min()
        print(f"{vname:<22} {g.mean():>10.3f} {g.std():>10.3f} "
              f"{asym:>10.3f} {maxdif:>14.3f}")
    print(f"{'='*70}")
    print(f"\nInterpretation:")
    print(f"  High L/R asymmetry = FiLM modulating left vs right differently")
    print(f"  Values far from 1.0 = FiLM actively scaling that limb's embedding")
    print(f"  Uniform gamma (~1.0 everywhere) = FiLM not contributing much")


if __name__ == "__main__":
    main()