"""
record_traj_isaac.py

Records a trajectory from an existing checkpoint in Isaac Gym.
No training — just loads checkpoint, runs policy, saves qpos + dof_pos.
Replay locally with replay_traj_mujoco.py.

Usage:
    python record_traj_isaac.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --variant_name robot_variant_3 \
        --num_steps 500 \
        --cmd_vx 0.5 \
        --out traj_variant3.pkl
"""

import isaacgym
from isaacgym import gymapi, gymutil

import argparse
import os
import sys
import json
import pickle
import numpy as np
import torch
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",         required=True)
    p.add_argument("--xml_path",           required=True)
    p.add_argument("--variants_metadata",  required=True)
    p.add_argument("--variant_name",       required=True,
                   help="Which variant to record, e.g. robot_variant_3")
    p.add_argument("--baseline", action="store_true", default=False,
               help="Use baseline cfg (no FiLM, no GCN)")
    p.add_argument("--num_steps",          type=int,   default=500)
    p.add_argument("--num_envs",           type=int,   default=512)
    p.add_argument("--cmd_vx",             type=float, default=0.5)
    p.add_argument("--cmd_vy",             type=float, default=0.0)
    p.add_argument("--cmd_yaw",            type=float, default=0.0)
    p.add_argument("--sim_device",         type=str,   default="cuda:0")
    p.add_argument("--rl_device",          type=str,   default="cuda:0")
    p.add_argument("--out",                type=str,   default="trajectory.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    from modular_policy.config import cfg
    from modular_policy.algos.ppo.runner import ModularRunner
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
    from legged_gym.utils.helpers import parse_sim_params, class_to_dict, set_seed

    # ── Load metadata to find variant index ──────────────────────────────
    with open(args.variants_metadata) as f:
        meta = json.load(f)
    variant_names = list(meta.keys())
    if args.variant_name not in variant_names:
        raise ValueError(f"Variant '{args.variant_name}' not in metadata. "
                         f"Available: {variant_names}")
    variant_idx = variant_names.index(args.variant_name)
    print(f"Recording variant '{args.variant_name}' (idx={variant_idx})")

    # ── cfg ───────────────────────────────────────────────────────────────
    cfg.PPO.NUM_ENVS               = args.num_envs
    cfg.MODEL.MAX_LIMBS            = 13
    cfg.MODEL.MAX_JOINTS           = 12
    if args.baseline:
        cfg.MODEL.GRAPH_ENCODING       = "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = False
    else:
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
    print(f"Env ready — num_envs={env.num_envs}")

    # ── Runner ────────────────────────────────────────────────────────────
    # Use a temp log dir (won't be written to)
    log_dir = f"/tmp/record_{datetime.now().strftime('%H%M%S')}"
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

    # ── Set command ───────────────────────────────────────────────────────
    runner.commands[:, 0] = args.cmd_vx
    runner.commands[:, 1] = args.cmd_vy
    runner.commands[:, 2] = args.cmd_yaw

    # ── Reset and get initial obs ─────────────────────────────────────────
    env.reset_idx(torch.arange(env.num_envs, device=runner.device))
    obs = runner._get_obs_normalized()

    # Find envs belonging to target variant
    env_ids = (env.env_variant_ids == variant_idx).nonzero(
        as_tuple=True)[0]
    ei = env_ids[0].item()   # just record first env of this variant
    print(f"Recording env {ei} (variant {variant_idx} = '{args.variant_name}')")

    # ── Record loop ───────────────────────────────────────────────────────
    traj = []
    runner.actor_critic.eval()

    with torch.no_grad():
        for step in range(args.num_steps):
            val, act, logp, dmv, dmmu = runner.agent.act(
                obs, unimal_ids=[0] * env.num_envs)

            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)

            obs, _, _, _ = runner._step(real_actions)
            obs = runner._normalize_obs(obs)

            # Record root pose + joint angles for env ei
            root  = env.root_states[ei].cpu().numpy()
            dof   = env.dof_pos[ei].cpu().numpy()

            traj.append({
                "xyz":     root[:3].copy(),       # world position
                "quat":    root[3:7].copy(),      # [x,y,z,w] Isaac Gym convention
                "dof_pos": dof.copy(),             # 12 joint angles
                "step":    step,
            })

            if step % 100 == 0:
                print(f"  step {step}/{args.num_steps}  "
                      f"height={root[2]:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "trajectory":    traj,
        "variant_name":  args.variant_name,
        "variant_idx":   variant_idx,
        "cmd":           [args.cmd_vx, args.cmd_vy, args.cmd_yaw],
        "num_steps":     args.num_steps,
        "checkpoint":    args.checkpoint,
        # Joint name order (for replay alignment)
        "dof_names":     env.dof_names,
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved {len(traj)} frames → {args.out}")
    print(f"  dof_names: {env.dof_names}")


if __name__ == "__main__":
    main()