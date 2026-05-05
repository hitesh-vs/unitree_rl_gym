"""
collect_tsne_multi_robot.py

Collects transformer input embeddings from a multi-robot checkpoint
using MuJoCo simulation (no Isaac Gym needed).

Runs each variant XML separately, collects obs_embed from mu_net,
tags each limb with its robot name and semantic role.
Output: *_embeds.npy and *_labels.pkl ready for plot_tsne.py

Usage:
    cd /home/sviswasam/dr/unitree_rl_gym
    python collect_tsne_multi_robot.py \
        --checkpoint output_multi_robot/Mar23_00-15-38/model_400.pt \
        --xml_dir /home/sviswasam/dr/ModuMorph/modular/unitree_g1_actual/xml \
        --out_prefix output_tsne/tsne_topological_multibot_iter400 \
        --graph_encoding topological \
        --steps_per_variant 30
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
import torch
import mujoco
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, REPO_ROOT)

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic
from modular_policy.algos.ppo.obs_builder import _BoxSpace, _DictSpace
from modular_policy.graphs.parser import MujocoGraphParser

# Reuse helpers from record_traj_modular
sys.path.insert(0, os.path.join(REPO_ROOT, "deploy/deploy_mujoco"))
from record_traj_modular import (
    MujocoObsBuilder, get_default_angles, get_kp_kd_arrays,
    ACTION_SCALE, SIM_DT, DECIMATION,
)

GAIT_PERIOD = 0.8
GAIT_OFFSET = 0.5


# ── Label builder ─────────────────────────────────────────────────────────────

def get_limb_labels(xml_path, robot_name, seq_len):
    tree      = ET.parse(xml_path)
    root      = tree.getroot()
    worldbody = root.find("worldbody")
    nodes, parents = [], {}

    def traverse(body, parent_name):
        name = body.attrib.get("name", f"body_{len(nodes)}")
        nodes.append(name)
        parents[name] = parent_name
        for child in body.findall("body"):
            traverse(child, name)

    for body in worldbody.findall("body"):
        traverse(body, None)

    labels = []
    for bname in nodes:
        n     = bname.lower()
        side  = ("left"  if "left"  in n else
                 "right" if "right" in n else "center")
        jtype = ("hip_pitch" if "hip_pitch" in n else
                 "hip_roll"  if "hip_roll"  in n else
                 "hip_yaw"   if "hip_yaw"   in n else
                 "knee"      if "knee"      in n else
                 "ankle"     if "ankle"     in n else
                 "root"      if ("pelvis" in n or "torso" in n) else
                 "other")
        labels.append({
            "robot":    robot_name,
            "body":     bname,
            "semantic": f"{side}_{jtype}",
        })

    n_pad = seq_len - len(labels)
    labels.extend([{"robot": "pad", "body": "pad", "semantic": "pad"}] * n_pad)
    return labels


# ── Build obs space ───────────────────────────────────────────────────────────

def build_obs_space(graph_encoding, prop_dim, ctx_dim):
    inf = float("inf")
    obs_spaces = {
        "proprioceptive":   _BoxSpace(-inf, inf, (prop_dim,)),
        "context":          _BoxSpace(-inf, inf, (ctx_dim,)),
        "obs_padding_mask": _BoxSpace(-inf, inf, (13,)),
        "act_padding_mask": _BoxSpace(-inf, inf, (13,)),
        "edges":            _BoxSpace(-inf, inf, (24,)),
        "traversals":       _BoxSpace(-inf, inf, (13,)),
        "SWAT_RE":          _BoxSpace(-inf, inf, (13, 13, 3)),
    }
    if graph_encoding != "none":
        feat_dim = (7  if graph_encoding == "onehot"
                    else 6 if graph_encoding == "topological"
                    else cfg.MODEL.RWSE_K)
        obs_spaces["graph_node_features"] = _BoxSpace(-inf, inf, (13, feat_dim))
        obs_spaces["graph_A_norm"]        = _BoxSpace(0., 1., (13, 13))

    return _DictSpace(obs_spaces)


# ── Collect embeddings for one variant ───────────────────────────────────────

def collect_variant(xml_path, robot_name, actor_critic,
                    ob_mean, ob_var, graph_encoding,
                    steps, device, command):
    """
    Run MuJoCo sim for one variant, collect obs_embed activations.
    Returns list of label dicts (one per limb per step, padding removed).
    Embeddings are captured in actor_critic.mu_net._tsne_buffer.
    """
    obs_builder = MujocoObsBuilder(xml_path, device)
    m = obs_builder.m
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)

    kp_arr, kd_arr = get_kp_kd_arrays(m)
    default_angles = get_default_angles(m)

    # Reset to default pose
    mujoco.mj_resetData(m, d)
    d.qpos[2]  = 0.8
    d.qpos[3]  = 1.0
    d.qpos[7:] = default_angles
    mujoco.mj_forward(m, d)

    last_action  = np.zeros(12, dtype=np.float32)
    target_pos   = default_angles.copy()
    episode_step = 0
    clipob       = 10.0

    labels_collected = []

    for sim_step in range(steps * DECIMATION):
        tau = ((target_pos - d.qpos[7:]) * kp_arr +
               (np.zeros(12)  - d.qvel[6:]) * kd_arr)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if sim_step % DECIMATION != 0:
            continue

        # Build obs
        obs, _, _ = obs_builder.build(d, command, episode_step, last_action)

        # Normalise proprioceptive
        prop = obs["proprioceptive"]
        if ob_mean is not None:
            prop = ((prop - ob_mean) / (ob_var + 1e-8).sqrt()).clamp(-clipob, clipob)
            obs["proprioceptive"] = prop

        # Forward pass — embedding captured by _tsne_buffer hook
        with torch.no_grad():
            actor_critic(obs)

        # Apply action
        act_mask    = obs_builder.act_padding_mask[0].bool()
        with torch.no_grad():
            val, pi, _, _, _, _ = actor_critic(obs)
        real_action = pi.loc[0, ~act_mask].numpy()
        target_pos  = real_action * ACTION_SCALE + default_angles
        last_action = real_action.copy()
        episode_step += 1

        # Build labels for this step
        seq_len = cfg.MODEL.MAX_LIMBS
        step_labels = get_limb_labels(xml_path, robot_name, seq_len)
        labels_collected.extend(step_labels)

    return labels_collected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--xml_dir",        required=True,
                        help="Dir with *_stripped.xml files")
    parser.add_argument("--out_prefix",     required=True,
                        help="Output prefix e.g. output_tsne/tsne_topo_iter400")
    parser.add_argument("--graph_encoding", default="topological",
                        choices=["none", "onehot", "topological", "rwse"])
    parser.add_argument("--steps_per_variant", type=int, default=30,
                        help="Policy steps to collect per variant")
    parser.add_argument("--cmd_vx",  type=float, default=0.5)
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    device = args.device
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────
    cfg.MODEL.GRAPH_ENCODING = args.graph_encoding
    cfg.MODEL.MAX_LIMBS      = 13
    cfg.MODEL.MAX_JOINTS     = 12
    cfg.PPO.NUM_ENVS         = 1
    cfg.PPO.BATCH_SIZE       = 1
    cfg.DETERMINISTIC        = True
    cfg.ENV.WALKERS          = ["g1"]

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    ckpt    = torch.load(args.checkpoint, map_location=device)
    ob_mean = ckpt.get("ob_mean", None)
    ob_var  = ckpt.get("ob_var",  None)
    if ob_mean is not None:
        ob_mean = ob_mean.to(device)
        ob_var  = ob_var.to(device)

    prop_dim      = ckpt["ob_mean"].shape[0]   # 338
    limb_obs_size = prop_dim // cfg.MODEL.MAX_LIMBS  # 26
    ctx_dim       = cfg.MODEL.MAX_LIMBS * cfg.MODEL.MORPH_CTX_DIM # 13 x 12

    print(f"prop_dim={prop_dim} limb_obs_size={limb_obs_size}")

    # ── Build model ───────────────────────────────────────────────────────
    obs_space    = build_obs_space(args.graph_encoding, prop_dim, ctx_dim)
    action_space = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    actor_critic = ActorCritic(obs_space, action_space)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()

    # Activate t-SNE buffer
    actor_critic.mu_net._tsne_buffer = {"embeds": [], "active": True}

    # ── Find variant XMLs ─────────────────────────────────────────────────
    xml_files = sorted([
        f for f in os.listdir(args.xml_dir)
        if f.endswith("_stripped.xml")
    ])
    print(f"\nFound {len(xml_files)} variants:")
    for f in xml_files:
        print(f"  {f}")

    command = np.array([args.cmd_vx, 0., 0.], dtype=np.float32)

    # ── Collect per variant ───────────────────────────────────────────────
    all_labels = []

    for xml_file in xml_files:
        robot_name = xml_file.replace("_stripped.xml", "")
        xml_path   = os.path.join(args.xml_dir, xml_file)
        print(f"\nCollecting {args.steps_per_variant} steps "
              f"for {robot_name} ...", flush=True)

        labels = collect_variant(
            xml_path     = xml_path,
            robot_name   = robot_name,
            actor_critic = actor_critic,
            ob_mean      = ob_mean,
            ob_var       = ob_var,
            graph_encoding = args.graph_encoding,
            steps        = args.steps_per_variant,
            device       = device,
            command      = command,
        )
        all_labels.extend(labels)
        print(f"  Done — {len(labels)} label entries")

    # ── Extract and flatten embeddings ────────────────────────────────────
    actor_critic.mu_net._tsne_buffer["active"] = False

    raw      = actor_critic.mu_net._tsne_buffer["embeds"]
    print(f"\nRaw buffer: {len(raw)} tensors, "
          f"each shape {raw[0].shape if raw else 'N/A'}")

    # Each entry: (seq_len, batch=1, d_model)
    # cat along batch dim → (seq_len, total_steps, d_model)
    combined    = torch.cat(raw, dim=1)
    seq_len, N_total, d = combined.shape
    flat_embeds = combined.permute(1, 0, 2).reshape(-1, d).numpy()

    print(f"flat_embeds: {flat_embeds.shape}")
    print(f"all_labels:  {len(all_labels)}")

    # Trim
    min_len     = min(len(flat_embeds), len(all_labels))
    flat_embeds = flat_embeds[:min_len]
    all_labels  = all_labels[:min_len]

    # Remove padding
    keep        = [i for i, l in enumerate(all_labels) if l["semantic"] != "pad"]
    flat_embeds = flat_embeds[keep]
    flat_labels = [all_labels[i] for i in keep]

    print(f"\nAfter removing padding: {len(flat_embeds)} embeddings")
    print(f"Robots:   {sorted(set(l['robot']    for l in flat_labels))}")
    print(f"Semantic: {sorted(set(l['semantic'] for l in flat_labels))}")

    # ── Save ─────────────────────────────────────────────────────────────
    np.save(f"{args.out_prefix}_embeds.npy", flat_embeds)
    with open(f"{args.out_prefix}_labels.pkl", "wb") as f:
        pickle.dump(flat_labels, f)

    print(f"\nSaved → {args.out_prefix}_embeds.npy")
    print(f"Saved → {args.out_prefix}_labels.pkl")
    print("\nNow run:")
    print(f"  python modular_policy/plot_tsne.py "
          f"--prefix {args.out_prefix} "
          f"--labels 'Topological multibot iter400'")


if __name__ == "__main__":
    main()