"""
deploy/deploy_mujoco/visualise_attention.py

Visualises per-layer, per-limb attention maps from the trained transformer
policy. Works with checkpoints trained with FIX_ATTENTION=True (context-driven
attention) or standard self-attention.

Produces one heatmap per transformer layer showing which limb tokens attend
to which other limb tokens. Rows = query limbs, Columns = key limbs.
High value = query limb strongly attends to key limb.

With FIX_ATTENTION=True:
  Q, K come from context (morphology) tokens.
  V comes from obs tokens.
  So the map shows "morphology-driven routing" — which limb types cluster
  together based on their morphology context.

Usage:
    python deploy/deploy_mujoco/visualise_attention.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --urdf_path resources/robots/g1_variants_wide/robot_variant_0.urdf \
        --graph_encoding rwse \
        --out_dir attention_maps/ \
        --variant_name robot_variant_0

    # Compare two variants side by side
    python deploy/deploy_mujoco/visualise_attention.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --urdf_path resources/robots/g1_variants_wide/robot_variant_0.urdf \
        --compare_urdf resources/robots/g1_variants_wide/robot_variant_9.urdf \
        --graph_encoding rwse \
        --out_dir attention_maps/ \
        --variant_name robot_variant_0 \
        --compare_name robot_variant_9
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic, Agent
from modular_policy.algos.ppo.obs_builder import _BoxSpace, _DictSpace

sys.path.insert(0, os.path.dirname(__file__))
from eval_zeroshot import ZeroShotObsBuilder, MORPH_CTX_DIM


# ── Limb label builder ────────────────────────────────────────────────────────

def get_limb_labels(xml_path, num_limbs, max_limbs):
    """Build short semantic labels for each limb position."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    labels = []
    for i in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is None:
            labels.append(f"body_{i}")
            continue
        n = name.lower()
        # Short label
        if   "pelvis"      in n: lbl = "pelvis"
        elif "hip_pitch"   in n: lbl = ("L" if "left" in n else "R") + "_hip_p"
        elif "hip_roll"    in n: lbl = ("L" if "left" in n else "R") + "_hip_r"
        elif "hip_yaw"     in n: lbl = ("L" if "left" in n else "R") + "_hip_y"
        elif "knee"        in n: lbl = ("L" if "left" in n else "R") + "_knee"
        elif "ankle_pitch" in n: lbl = ("L" if "left" in n else "R") + "_ank_p"
        elif "ankle_roll"  in n: lbl = ("L" if "left" in n else "R") + "_ank_r"
        elif "ankle"       in n: lbl = ("L" if "left" in n else "R") + "_ankle"
        else:                    lbl = name[:12]
        labels.append(lbl)
        if len(labels) >= num_limbs:
            break

    # Pad
    while len(labels) < max_limbs:
        labels.append(f"pad_{len(labels)}")
    return labels[:max_limbs]


# ── Build a dummy obs for one robot ──────────────────────────────────────────

def build_dummy_obs(xml_path, urdf_path, device, num_dof=12):
    """
    Build a minimal obs dict with zero kinematics but correct context/graph.
    We only care about context and graph features for the attention map —
    the proprioceptive tokens are zeroed so the attention pattern reflects
    purely the morphology conditioning.
    """
    obs_builder = ZeroShotObsBuilder(
        xml_path, device, urdf_path=urdf_path, num_dof=num_dof)

    prop_dim = cfg.MODEL.MAX_LIMBS * obs_builder.limb_obs_size
    obs = {
        "proprioceptive":   torch.zeros(1, prop_dim, device=device),
        "context":          obs_builder.context,
        "edges":            obs_builder.edges,
        "traversals":       obs_builder.traversals,
        "SWAT_RE":          obs_builder.swat_re,
        "obs_padding_mask": obs_builder.obs_padding_mask,
        "act_padding_mask": obs_builder.act_padding_mask,
    }
    if obs_builder.graph_enc != "none":
        obs["graph_node_features"] = obs_builder.graph_node_features
        obs["graph_A_norm"]        = obs_builder.graph_A_norm

    return obs, obs_builder.num_limbs


# ── Extract attention maps ────────────────────────────────────────────────────

def get_attention_maps(actor_critic, obs, device):
    """
    Run a forward pass with return_attention=True.
    Returns list of attention maps, one per transformer layer.
    Each map: (batch=1, num_limbs, num_limbs) after masking padding.
    """
    actor_critic.eval()
    with torch.no_grad():
        _, _, v_attn, mu_attn, _, _ = actor_critic(
            obs, return_attention=True, compute_val=False,
            unimal_ids=[0])
    # mu_attn: list of (batch, tgt_seq, src_seq) per layer
    # tgt_seq = src_seq = max_limbs
    return mu_attn   # use actor (mu_net) attention


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_attention(attn_maps, limb_labels, num_limbs, title, out_path,
                   obs_mask=None):
    """
    Plot one figure with one subplot per transformer layer.
    attn_maps: list of (1, max_limbs, max_limbs) tensors
    """
    n_layers = len(attn_maps)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    # Only show the non-padded limbs
    L = num_limbs
    row_labels = limb_labels[:L]
    col_labels = limb_labels[:L]

    for layer_idx, (ax, attn) in enumerate(zip(axes, attn_maps)):
        # attn: (1, max_limbs, max_limbs) — crop to real limbs
        a = attn[0, :L, :L].cpu().numpy()   # (L, L)

        im = ax.imshow(a, cmap="Blues", vmin=0, vmax=a.max(),
                       aspect="auto", interpolation="nearest")

        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=9)
        ax.set_xlabel("Key limb", fontsize=8)
        if layer_idx == 0:
            ax.set_ylabel("Query limb", fontsize=8)

        # Annotate cells with value
        for i in range(L):
            for j in range(L):
                ax.text(j, i, f"{a[i,j]:.2f}", ha="center", va="center",
                        fontsize=5, color="black" if a[i,j] < a.max()*0.6 else "white")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_attention_comparison(attn_maps_a, attn_maps_b,
                               limb_labels, num_limbs,
                               name_a, name_b, out_path):
    """
    Plot two robots side-by-side per layer, plus difference map.
    Shows where attention patterns change across morphology variants.
    """
    n_layers = len(attn_maps_a)
    L        = num_limbs
    fig, axes = plt.subplots(n_layers, 3,
                             figsize=(15, 4 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    labels = limb_labels[:L]

    for li in range(n_layers):
        a = attn_maps_a[li][0, :L, :L].cpu().numpy()
        b = attn_maps_b[li][0, :L, :L].cpu().numpy()
        d = b - a   # positive = B attends more, negative = A attends more

        vmax = max(a.max(), b.max())
        dmax = np.abs(d).max()

        for col, (data, title, cmap, vmin, vm) in enumerate([
            (a, name_a,    "Blues", 0,    vmax),
            (b, name_b,    "Blues", 0,    vmax),
            (d, "B - A",   "RdBu",  -dmax, dmax),
        ]):
            ax = axes[li, col]
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vm,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(L)); ax.set_yticks(range(L))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_title(f"L{li+1}: {title}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--xml_path",      required=True,
                        help="Stripped topology XML (same for all G1 variants)")
    parser.add_argument("--urdf_path",     required=True,
                        help="Variant URDF for morphology context")
    parser.add_argument("--compare_urdf",  default=None,
                        help="Second variant URDF to compare against")
    parser.add_argument("--variant_name",  default="variant_a")
    parser.add_argument("--compare_name",  default="variant_b")
    parser.add_argument("--graph_encoding", default="rwse",
                        choices=["none", "onehot", "topological", "rwse"])
    parser.add_argument("--num_dof",  type=int, default=12)
    parser.add_argument("--out_dir",  default="attention_maps")
    parser.add_argument("--device",   default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Config ────────────────────────────────────────────────────────────
    cfg.MODEL.GRAPH_ENCODING = args.graph_encoding
    cfg.MODEL.MAX_LIMBS      = 13
    cfg.MODEL.MAX_JOINTS     = 12
    cfg.PPO.NUM_ENVS         = 1
    cfg.PPO.BATCH_SIZE       = 1
    cfg.ENV.WALKERS          = ["g1"]

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    has_film = any("film_generator" in k
                   for k in ckpt["model_state_dict"].keys())
    cfg.MODEL.TRANSFORMER.USE_FILM = has_film
    print(f"  FiLM: {has_film}")

    limb_obs_size = 26 if args.graph_encoding != "none" else 32
    prop_dim      = cfg.MODEL.MAX_LIMBS * limb_obs_size
    ctx_dim       = cfg.MODEL.MAX_LIMBS * MORPH_CTX_DIM
    inf           = float("inf")

    obs_spaces = {
        "proprioceptive":   _BoxSpace(-inf, inf, (prop_dim,)),
        "context":          _BoxSpace(-inf, inf, (ctx_dim,)),
        "obs_padding_mask": _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS,)),
        "act_padding_mask": _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS,)),
        "edges":            _BoxSpace(-inf, inf, (cfg.MODEL.MAX_JOINTS * 2,)),
        "traversals":       _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS,)),
        "SWAT_RE":          _BoxSpace(-inf, inf,
                                     (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3)),
    }
    if args.graph_encoding != "none":
        fd = 7 if args.graph_encoding == "onehot" else 6
        obs_spaces["graph_node_features"] = _BoxSpace(
            -inf, inf, (cfg.MODEL.MAX_LIMBS, fd))
        obs_spaces["graph_A_norm"] = _BoxSpace(
            0., 1., (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS))

    observation_space = _DictSpace(obs_spaces)
    action_space      = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    actor_critic = ActorCritic(observation_space, action_space).to(device)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()

    # ── Build obs for variant A ───────────────────────────────────────────
    print(f"\nBuilding obs for: {args.variant_name}")
    obs_a, num_limbs = build_dummy_obs(
        args.xml_path, args.urdf_path, device, args.num_dof)

    limb_labels = get_limb_labels(
        args.xml_path, num_limbs, cfg.MODEL.MAX_LIMBS)
    print(f"  Limb labels: {limb_labels[:num_limbs]}")

    # ── Get attention maps for variant A ──────────────────────────────────
    print(f"  Running forward pass...")
    attn_a = get_attention_maps(actor_critic, obs_a, device)
    n_layers = len(attn_a)
    print(f"  Got {n_layers} attention maps, "
          f"each shape {attn_a[0].shape}")

    out_a = os.path.join(args.out_dir,
                         f"attn_{args.variant_name}.png")
    plot_attention(
        attn_a, limb_labels, num_limbs,
        title=f"Attention maps — {args.variant_name}",
        out_path=out_a)

    # ── Optional comparison ───────────────────────────────────────────────
    if args.compare_urdf:
        print(f"\nBuilding obs for: {args.compare_name}")
        obs_b, _ = build_dummy_obs(
            args.xml_path, args.compare_urdf, device, args.num_dof)

        attn_b = get_attention_maps(actor_critic, obs_b, device)

        out_b = os.path.join(args.out_dir,
                             f"attn_{args.compare_name}.png")
        plot_attention(
            attn_b, limb_labels, num_limbs,
            title=f"Attention maps — {args.compare_name}",
            out_path=out_b)

        out_cmp = os.path.join(args.out_dir,
                               f"attn_compare_{args.variant_name}_vs_"
                               f"{args.compare_name}.png")
        plot_attention_comparison(
            attn_a, attn_b, limb_labels, num_limbs,
            args.variant_name, args.compare_name, out_cmp)

        # Print the biggest differences
        print(f"\nTop attention differences ({args.compare_name} - {args.variant_name}):")
        for li, (a, b) in enumerate(zip(attn_a, attn_b)):
            diff = (b - a)[0, :num_limbs, :num_limbs].cpu().numpy()
            idx  = np.unravel_index(np.abs(diff).argmax(), diff.shape)
            print(f"  Layer {li+1}: biggest change at "
                  f"({limb_labels[idx[0]]} → {limb_labels[idx[1]]}) "
                  f"Δ={diff[idx]:.3f}")

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()