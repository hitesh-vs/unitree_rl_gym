"""
deploy/deploy_mujoco/record_traj_all_variants.py

Runs trajectory recording for all variants in variants_metadata.json
and prints a summary table of episode stats per variant.

Changes vs original:
  1. Uses ZeroShotObsBuilder (12-dim context) instead of MujocoObsBuilder.
  2. Auto-detects FiLM from checkpoint keys.
  3. Per-variant ob_mean/ob_var — checkpoint saves (V, prop_dim), each
     variant gets its own row matched by name from --train_variants_metadata.
     Unseen/held-out variants get the mean across all training variants.
  4. Added --graph_encoding rwse and --train_variants_metadata args.

Usage:
    # Test on training variants (exact stats per variant)
    python deploy/deploy_mujoco/record_traj_all_variants.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --train_variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --base_xml /path/to/g1_12dof_stripped.xml \
        --out_dir trajectories/film_train \
        --duration 10.0 --cmd_vx 0.5 --graph_encoding rwse

    # Test on held-out variants (averaged stats for unseen robots)
    python deploy/deploy_mujoco/record_traj_all_variants.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --variants_metadata resources/robots/g1_variants_heldout/variants_metadata.json \
        --train_variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --base_xml /path/to/g1_12dof_stripped.xml \
        --out_dir trajectories/film_heldout \
        --duration 10.0 --cmd_vx 0.5 --graph_encoding rwse
"""

import os
import sys
import json
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

sys.path.insert(0, os.path.dirname(__file__))
from eval_zeroshot import ZeroShotObsBuilder, MORPH_CTX_DIM
from record_traj_modular import (
    get_kp_kd_arrays, get_default_angles,
    quat_rotate_inverse_np,
    SIM_DT, DECIMATION, POLICY_DT, ACTION_SCALE,
)

import mujoco


def run_variant(variant_name, xml_path, urdf_path, base_height_target,
                actor_critic, agent, ob_mean, ob_var,
                command, duration, graph_encoding, device,
                output_file):
    clipob = 10.0

    def normalize_obs(obs_dict):
        prop = obs_dict["proprioceptive"]
        obs_dict["proprioceptive"] = (
            (prop - ob_mean) / (ob_var + 1e-8).sqrt()
        ).clamp(-clipob, clipob)
        return obs_dict

    obs_builder = ZeroShotObsBuilder(
        xml_path, device, urdf_path=urdf_path, num_dof=12)

    m = obs_builder.m
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)

    kp_arr, kd_arr = get_kp_kd_arrays(m)
    default_angles = get_default_angles(m)

    mujoco.mj_resetData(m, d)
    d.qpos[2]  = base_height_target + 0.02
    d.qpos[3]  = 1.0
    d.qpos[7:] = default_angles
    mujoco.mj_forward(m, d)

    max_sim_steps   = int(duration / SIM_DT)
    last_action     = np.zeros(12, dtype=np.float32)
    target_dof_pos  = default_angles.copy()
    episode_step    = 0
    sim_step        = 0
    total_reward    = 0.0
    ep_len          = 0
    episodes        = 0
    episode_rewards = []
    trajectory      = []

    h_lo = base_height_target * 0.5
    h_hi = base_height_target * 1.5

    while sim_step < max_sim_steps:
        tau = ((target_dof_pos - d.qpos[7:]) * kp_arr +
               (np.zeros(12)   - d.qvel[6:]) * kd_arr)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        sim_step += 1

        trajectory.append({
            "qpos":         d.qpos.copy(),
            "qvel":         d.qvel.copy(),
            "action":       last_action.copy(),
            "target_pos":   target_dof_pos.copy(),
            "time":         d.time,
            "episode_step": episode_step,
        })

        if sim_step % DECIMATION != 0:
            continue

        obs, _, _ = obs_builder.build(d, command, episode_step, last_action)
        # Debug: print raw prop stats before normalisation
        if episode_step == 5:
            raw = obs["proprioceptive"][0]
            print(f"  raw prop: mean={raw.mean():.3f} std={raw.std():.3f} "
                  f"min={raw.min():.3f} max={raw.max():.3f}")
            print(f"  ob_mean:  mean={ob_mean.mean():.3f} std={ob_mean.std():.3f}")
            print(f"  ob_var:   mean={ob_var.mean():.3f}")
        obs = normalize_obs(obs)

        with torch.no_grad():
            _, act, _, _, _ = agent.act(obs, unimal_ids=[0])

        act_mask       = obs_builder.act_padding_mask[0].bool()
        real_action    = act[0, ~act_mask].cpu().numpy()
        target_dof_pos = real_action * ACTION_SCALE + default_angles
        last_action    = real_action.copy()
        episode_step  += 1

        r = 0.15 * POLICY_DT
        base_vel_body = quat_rotate_inverse_np(d.qpos[3:7], d.qvel[:3])
        lin_err = np.sum((command[:2] - base_vel_body[:2])**2)
        r += 1.0 * np.exp(-lin_err / 0.25) * POLICY_DT
        total_reward += r
        ep_len       += 1

        height = d.qpos[2]
        w, x, y, z = d.qpos[3:7]
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        fell  = (height < h_lo or height > h_hi or
                 abs(pitch) > 1.0 or abs(roll) > 0.8)

        if fell:
            episode_rewards.append({
                "episode":      episodes + 1,
                "return":       round(total_reward, 3),
                "length_secs":  round(ep_len * POLICY_DT, 2),
                "termination":  "fall",
            })
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

    if ep_len > 0:
        episode_rewards.append({
            "episode":      episodes + 1,
            "return":       round(total_reward, 3),
            "length_secs":  round(ep_len * POLICY_DT, 2),
            "termination":  "timeout",
        })

    with open(output_file, "wb") as f:
        pickle.dump({
            "trajectory":         trajectory,
            "episode_rewards":    episode_rewards,
            "variant_name":       variant_name,
            "command":            command.tolist(),
            "xml_path":           xml_path,
            "base_height_target": base_height_target,
            "graph_encoding":     graph_encoding,
            "sim_dt":             SIM_DT,
            "policy_dt":          POLICY_DT,
        }, f)

    return episode_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",               required=True)
    parser.add_argument("--variants_metadata",        required=True,
                        help="Metadata to evaluate on (train or heldout)")
    parser.add_argument("--train_variants_metadata",  required=True,
                        help="Metadata used during training — used to match "
                             "variant names to ob_mean/ob_var row indices")
    parser.add_argument("--out_dir",    default="trajectories")
    parser.add_argument("--duration",   type=float, default=10.0)
    parser.add_argument("--cmd_vx",     type=float, default=0.5)
    parser.add_argument("--cmd_vy",     type=float, default=0.0)
    parser.add_argument("--cmd_yaw",    type=float, default=0.0)
    parser.add_argument("--graph_encoding", default="rwse",
                        choices=["none", "onehot", "topological", "rwse"])
    parser.add_argument("--base_xml",   default=None,
                        help="Override xml path in metadata")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device  = torch.device(args.device)
    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)

    cfg.MODEL.GRAPH_ENCODING = args.graph_encoding
    cfg.MODEL.MAX_LIMBS      = 13
    cfg.MODEL.MAX_JOINTS     = 12
    cfg.PPO.NUM_ENVS         = 1
    cfg.PPO.BATCH_SIZE       = 1
    cfg.ENV.WALKERS          = ["g1"]

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    has_film = any("film_generator" in k
                   for k in ckpt["model_state_dict"].keys())
    cfg.MODEL.TRANSFORMER.USE_FILM = has_film
    print(f"  FiLM: {has_film}")

    # ── Per-variant ob_mean/ob_var ────────────────────────────────────────
    # Checkpoint saves (V, prop_dim) — one row per training variant in order.
    # We match variant names from train_variants_metadata to get the right row.
    ob_mean_all = ckpt.get("ob_mean")
    ob_var_all  = ckpt.get("ob_var")

    if ob_mean_all is None:
        limb_obs_size = 26 if args.graph_encoding != "none" else 32
        prop_dim      = cfg.MODEL.MAX_LIMBS * limb_obs_size
        ob_mean_all   = torch.zeros(1, prop_dim)
        ob_var_all    = torch.ones(1,  prop_dim)

    if ob_mean_all.dim() == 1:
        ob_mean_all = ob_mean_all.unsqueeze(0)
        ob_var_all  = ob_var_all.unsqueeze(0)

    ob_mean_all = ob_mean_all.to(device)
    ob_var_all  = ob_var_all.to(device)

    # Fallback stats for unseen variants: mean across all training variants
    ob_mean_fallback = ob_mean_all.mean(0)
    ob_var_fallback  = ob_var_all.mean(0)

    print(f"  ob_mean shape: {ob_mean_all.shape}  "
          f"(V={ob_mean_all.shape[0]} variants)")

    # Build name → row-index map from training metadata
    with open(args.train_variants_metadata) as f:
        train_meta = json.load(f)
    train_variant_order = list(train_meta.keys())
    print(f"  Training variant order ({len(train_variant_order)}): "
          f"{train_variant_order}")

    # ── Build obs spaces ──────────────────────────────────────────────────
    limb_obs_size = 26 if args.graph_encoding != "none" else 32
    prop_dim      = cfg.MODEL.MAX_LIMBS * limb_obs_size
    ctx_dim       = cfg.MODEL.MAX_LIMBS * MORPH_CTX_DIM   # 13 * 12 = 156
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
        obs_spaces["graph_A_norm"]        = _BoxSpace(
            0., 1., (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS))

    observation_space = _DictSpace(obs_spaces)
    action_space      = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    actor_critic = ActorCritic(observation_space, action_space).to(device)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()
    agent = Agent(actor_critic)

    # ── Load eval metadata ────────────────────────────────────────────────
    with open(args.variants_metadata) as f:
        meta = json.load(f)

    # ── Run each variant ──────────────────────────────────────────────────
    summary = {}

    for variant_name, variant_meta in meta.items():
        xml_path  = args.base_xml or variant_meta.get("xml")
        urdf_path = variant_meta.get("urdf")
        bht       = variant_meta.get("base_height_target", 0.78)

        # Look up per-variant normalisation stats
        if variant_name in train_variant_order:
            v_idx  = train_variant_order.index(variant_name)
            ob_mean = ob_mean_all[v_idx]
            ob_var  = ob_var_all[v_idx]
            stats_src = f"training row {v_idx}"
        else:
            ob_mean = ob_mean_fallback
            ob_var  = ob_var_fallback
            stats_src = "mean of all training variants (unseen)"

        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}  bht={bht}  stats={stats_src}")
        print(f"  xml={os.path.basename(xml_path or '')}  "
              f"urdf={os.path.basename(urdf_path or '')}")
        print(f"{'='*60}")

        out_file = os.path.join(args.out_dir, f"traj_{variant_name}.pkl")

        t0 = time.time()
        try:
            episode_rewards = run_variant(
                variant_name        = variant_name,
                xml_path            = xml_path,
                urdf_path           = urdf_path,
                base_height_target  = bht,
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
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            summary[variant_name] = {"episodes": 0, "error": str(e)}
            continue

        elapsed = time.time() - t0

        if episode_rewards:
            returns = [e["return"]      for e in episode_rewards]
            lengths = [e["length_secs"] for e in episode_rewards]
            falls   = sum(1 for e in episode_rewards
                          if e["termination"] == "fall")
            summary[variant_name] = {
                "episodes":    len(episode_rewards),
                "mean_return": round(float(np.mean(returns)), 3),
                "mean_ep_len": round(float(np.mean(lengths)), 2),
                "max_ep_len":  round(float(np.max(lengths)),  2),
                "fall_rate":   round(falls / len(episode_rewards), 3),
                "stats_src":   stats_src,
                "pkl":         out_file,
            }
            print(f"  eps={len(episode_rewards)}  "
                  f"mean_len={summary[variant_name]['mean_ep_len']:.1f}s  "
                  f"fall_rate={summary[variant_name]['fall_rate']:.2f}")
        else:
            summary[variant_name] = {"episodes": 0, "pkl": out_file}

        print(f"  Done in {elapsed:.1f}s → {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"{'Variant':<22} {'Eps':>4} {'MeanRet':>8} "
          f"{'MeanLen(s)':>11} {'MaxLen(s)':>10} {'FallRate':>9}")
    print(f"{'-'*76}")
    for name, s in summary.items():
        if s.get("episodes", 0) > 0:
            print(f"{name:<22} {s['episodes']:>4} {s['mean_return']:>8.2f} "
                  f"{s['mean_ep_len']:>11.1f} {s['max_ep_len']:>10.1f} "
                  f"{s['fall_rate']:>9.3f}")
        else:
            print(f"{name:<22}    0  "
                  f"{'ERROR: '+s.get('error','') if 'error' in s else '(no data)'}")
    print(f"{'='*76}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()