"""
deploy/deploy_mujoco/record_traj_all_variants.py

Runs trajectory recording for all variants in variants_metadata.json
and prints a summary table of episode stats per variant.

Usage:
    cd /home/sviswasam/dr/unitree_rl_gym
    python deploy/deploy_mujoco/record_traj_all_variants.py \
        --checkpoint output_walk_isaac_success/Mar21_18-51-07/model_400.pt \
        --variants_metadata resources/robots/g1_variants/variants_metadata.json \
        --out_dir trajectories/ \
        --duration 10.0 \
        --cmd_vx 0.5 \
        --graph_encoding topological
"""

import os
import sys
import json
import math
import time
import pickle
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic, Agent
from modular_policy.algos.ppo.obs_builder import _BoxSpace, _DictSpace

# Import the per-step helpers from the single-variant script
sys.path.insert(0, os.path.dirname(__file__))
from record_traj_modular import (
    MujocoObsBuilder, get_kp_kd_arrays, get_default_angles,
    quat_rotate_inverse_np,
    SIM_DT, DECIMATION, POLICY_DT, ACTION_SCALE,
)

import mujoco


def run_variant(variant_name, xml_path, base_height_target,
                actor_critic, agent, ob_mean, ob_var,
                command, duration, graph_encoding, device,
                output_file):
    """Run one variant and save trajectory. Returns episode stats."""

    clipob = 10.0

    def normalize_obs(obs_dict):
        prop = obs_dict["proprioceptive"]
        prop = ((prop - ob_mean) / (ob_var + 1e-8).sqrt()).clamp(-clipob, clipob)
        obs_dict["proprioceptive"] = prop
        return obs_dict

    obs_builder = MujocoObsBuilder(xml_path, device)

    m = obs_builder.m
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)

    kp_arr, kd_arr = get_kp_kd_arrays(m)
    default_angles = get_default_angles(m)

    # Reset
    mujoco.mj_resetData(m, d)
    d.qpos[2]  = base_height_target + 0.02   # spawn slightly above target
    d.qpos[3]  = 1.0
    d.qpos[7:] = default_angles
    mujoco.mj_forward(m, d)

    max_sim_steps  = int(duration / SIM_DT)
    last_action    = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    episode_step   = 0
    sim_step       = 0
    total_reward   = 0.0
    ep_len         = 0
    episodes       = 0
    episode_rewards = []
    trajectory      = []

    # Height bounds per variant
    h_lo = base_height_target * 0.5
    h_hi = base_height_target * 1.5

    while sim_step < max_sim_steps:
        tau = (target_dof_pos - d.qpos[7:]) * kp_arr + \
              (np.zeros(12)   - d.qvel[6:]) * kd_arr
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        sim_step += 1

        trajectory.append({
            "qpos":        d.qpos.copy(),
            "qvel":        d.qvel.copy(),
            "action":      last_action.copy(),
            "target_pos":  target_dof_pos.copy(),
            "time":        d.time,
            "episode_step": episode_step,
        })

        if sim_step % DECIMATION != 0:
            continue

        obs, _, _ = obs_builder.build(d, command, episode_step, last_action)
        obs = normalize_obs(obs)

        with torch.no_grad():
            _, act, _, _, _ = agent.act(obs, unimal_ids=[0])

        act_mask       = obs_builder.act_padding_mask[0].bool()
        real_action    = act[0, ~act_mask].cpu().numpy()
        target_dof_pos = real_action * ACTION_SCALE + default_angles
        last_action    = real_action.copy()
        episode_step  += 1

        # Reward tracking
        r = 0.15 * POLICY_DT
        base_vel_body = quat_rotate_inverse_np(d.qpos[3:7], d.qvel[:3])
        lin_err = np.sum((command[:2] - base_vel_body[:2])**2)
        r += 1.0 * np.exp(-lin_err / 0.25) * POLICY_DT
        total_reward += r
        ep_len       += 1

        # Termination
        height = d.qpos[2]
        w, x, y, z = d.qpos[3:7]
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))

        fell = height < h_lo or height > h_hi or abs(pitch) > 1.0 or abs(roll) > 0.8

        if fell:
            episode_rewards.append({
                "episode":      episodes + 1,
                "return":       round(total_reward, 3),
                "length_secs":  round(ep_len * POLICY_DT, 2),
                "termination":  "fall",
            })
            # Reset
            mujoco.mj_resetData(m, d)
            d.qpos[2]  = base_height_target + 0.02
            d.qpos[3]  = 1.0
            d.qpos[7:] = default_angles
            mujoco.mj_forward(m, d)
            last_action    = np.zeros(12, dtype=np.float32)
            target_dof_pos = default_angles.copy()
            episode_step   = 0
            total_reward   = 0.0
            ep_len         = 0
            episodes      += 1

    # Final episode
    if ep_len > 0:
        episode_rewards.append({
            "episode":      episodes + 1,
            "return":       round(total_reward, 3),
            "length_secs":  round(ep_len * POLICY_DT, 2),
            "termination":  "timeout",
        })

    # Save
    output = {
        "trajectory":      trajectory,
        "episode_rewards": episode_rewards,
        "variant_name":    variant_name,
        "command":         command.tolist(),
        "xml_path":        xml_path,
        "base_height_target": base_height_target,
        "graph_encoding":  graph_encoding,
        "sim_dt":          SIM_DT,
        "policy_dt":       POLICY_DT,
    }
    with open(output_file, "wb") as f:
        pickle.dump(output, f)

    return episode_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--variants_metadata", required=True)
    parser.add_argument("--out_dir",           default="trajectories")
    parser.add_argument("--duration",   type=float, default=10.0)
    parser.add_argument("--cmd_vx",     type=float, default=0.5)
    parser.add_argument("--cmd_vy",     type=float, default=0.0)
    parser.add_argument("--cmd_yaw",    type=float, default=0.0)
    parser.add_argument("--graph_encoding", default="topological",
                        choices=["none", "onehot", "topological"])
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device  = torch.device(args.device)
    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)

    # ── Config ────────────────────────────────────────────────────────────
    cfg.MODEL.GRAPH_ENCODING = args.graph_encoding
    cfg.MODEL.MAX_LIMBS      = 13
    cfg.MODEL.MAX_JOINTS     = 12
    cfg.PPO.NUM_ENVS         = 1
    cfg.PPO.BATCH_SIZE       = 1
    cfg.ENV.WALKERS          = ["g1"]

    # ── Load metadata ─────────────────────────────────────────────────────
    with open(args.variants_metadata) as f:
        meta = json.load(f)

    # ── Load checkpoint once — shared across all variants ─────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Build spaces from first variant XML to get dims
    first_xml = list(meta.values())[0]["xml"]
    tmp_builder = MujocoObsBuilder(first_xml, device)
    prop_dim    = cfg.MODEL.MAX_LIMBS * tmp_builder.limb_obs_size
    ctx_dim     = int(tmp_builder.context.shape[1])
    inf         = float("inf")

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
        obs_spaces["graph_A_norm"]        = _BoxSpace(
            0., 1., (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS))

    observation_space = _DictSpace(obs_spaces)
    action_space      = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    actor_critic = ActorCritic(observation_space, action_space).to(device)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()
    agent = Agent(actor_critic)

    ob_mean = ckpt.get("ob_mean", torch.zeros(prop_dim)).to(device)
    ob_var  = ckpt.get("ob_var",  torch.ones(prop_dim)).to(device)

    # ── Run each variant ──────────────────────────────────────────────────
    summary = {}

    for variant_name, variant_meta in meta.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}  "
              f"base_height_target={variant_meta['base_height_target']}")
        print(f"{'='*60}")

        out_file = os.path.join(args.out_dir, f"traj_{variant_name}.pkl")

        t0 = time.time()
        episode_rewards = run_variant(
            variant_name        = variant_name,
            xml_path            = variant_meta["xml"],
            base_height_target  = variant_meta["base_height_target"],
            actor_critic        = actor_critic,
            agent               = agent,
            ob_mean             = ob_mean,
            ob_var              = ob_var,
            command             = command,
            duration            = args.duration,
            graph_encoding      = args.graph_encoding,
            device              = device,
            output_file         = out_file,
        )
        elapsed = time.time() - t0

        if episode_rewards:
            returns = [e["return"]       for e in episode_rewards]
            lengths = [e["length_secs"]  for e in episode_rewards]
            summary[variant_name] = {
                "episodes":       len(episode_rewards),
                "mean_return":    round(float(np.mean(returns)), 3),
                "mean_ep_len":    round(float(np.mean(lengths)), 2),
                "max_ep_len":     round(float(np.max(lengths)), 2),
                "max_return":     round(float(np.max(returns)), 3),
                "pkl":            out_file,
            }
        else:
            summary[variant_name] = {"episodes": 0, "pkl": out_file}

        print(f"Done in {elapsed:.1f}s — saved to {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Variant':<22} {'Eps':>4} {'MeanRet':>8} "
          f"{'MeanLen(s)':>11} {'MaxLen(s)':>10}")
    print(f"{'-'*70}")
    for name, s in summary.items():
        if s["episodes"] > 0:
            print(f"{name:<22} {s['episodes']:>4} {s['mean_return']:>8.2f} "
                  f"{s['mean_ep_len']:>11.1f} {s['max_ep_len']:>10.1f}")
        else:
            print(f"{name:<22}    0  (no data)")
    print(f"{'='*70}")

    # Save summary JSON
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()