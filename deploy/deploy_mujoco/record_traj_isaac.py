"""
record_traj_isaac.py

Records a trajectory from an existing checkpoint in Isaac Gym.
Also runs multiple rollouts to measure avg episode length as a metric.

Usage:
    python record_traj_isaac.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --variant_name robot_variant_4 \
        --num_steps 500 \
        --num_eval_rollouts 10 \
        --cmd_vx 0.5 \
        --out traj_film_variant4.pkl
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
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--xml_path",            required=True)
    p.add_argument("--variants_metadata",   required=True)
    p.add_argument("--variant_name",        required=True)
    p.add_argument("--baseline",            action="store_true", default=False)
    p.add_argument("--ood",                 action="store_true", default=False,
                   help="OOD mode — use mean obs stats, print context debug")
    p.add_argument("--num_steps",           type=int,   default=500)
    p.add_argument("--num_eval_rollouts",   type=int,   default=10)
    p.add_argument("--max_ep_steps",        type=int,   default=1000)
    p.add_argument("--num_envs",            type=int,   default=512)
    p.add_argument("--cmd_vx",             type=float, default=0.5)
    p.add_argument("--cmd_vy",             type=float, default=0.0)
    p.add_argument("--cmd_yaw",            type=float, default=0.0)
    p.add_argument("--sim_device",         type=str,   default="cuda:0")
    p.add_argument("--rl_device",          type=str,   default="cuda:0")
    p.add_argument("--out",                type=str,   default="trajectory.pkl")
    return p.parse_args()


def run_eval_rollouts(runner, env, variant_idx, num_rollouts,
                      max_ep_steps, device, ood_mode=False):
    env_ids_variant = (env.env_variant_ids == variant_idx).nonzero(
        as_tuple=True)[0]
    n_variant_envs = len(env_ids_variant)
    print(f"\n[Eval] {num_rollouts} rollouts across "
          f"{n_variant_envs} envs of variant {variant_idx}...")

    episode_lengths  = []
    ep_step_counters = torch.zeros(env.num_envs, dtype=torch.long,  device=device)

    env.reset_idx(torch.arange(env.num_envs, device=device))
    runner.commands[:, 0] = runner.commands[:, 0] * 0 + 0.5
    obs = runner._get_obs_normalized()

    # ── Context debug — printed once at the very start ────────────────────
    ei0 = env_ids_variant[0].item()
    ctx = obs["context"][ei0]
    print(f"\n[Context debug] variant_idx={variant_idx}  ood_mode={ood_mode}")
    print(f"  context shape : {ctx.shape}")
    print(f"  context mean  : {ctx.mean():.4f}")
    print(f"  context std   : {ctx.std():.4f}")
    print(f"  context min   : {ctx.min():.4f}")
    print(f"  context max   : {ctx.max():.4f}")
    print(f"  first 12 vals (limb 0): {ctx[:12].cpu().numpy().round(4)}")
    print(f"  next  12 vals (limb 1): {ctx[12:24].cpu().numpy().round(4)}")

    # ── Also print prop obs for comparison ────────────────────────────────
    prop = obs["proprioceptive"][ei0]
    print(f"  prop mean={prop.mean():.3f}  std={prop.std():.3f}  "
          f"min={prop.min():.3f}  max={prop.max():.3f}")

    collected = 0
    step      = 0
    ctx_printed = False

    runner.actor_critic.eval()
    with torch.no_grad():
        while collected < num_rollouts * n_variant_envs:
            val, act, logp, dmv, dmmu = runner.agent.act(
                obs, unimal_ids=[0] * env.num_envs)

            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)

            obs, _, dones, _ = runner._step(real_actions)
            obs = runner._normalize_obs(obs)

            ep_step_counters += 1
            step += 1

            # Height termination debug for first 4 steps
            if step < 5:
                for ei_dbg in env_ids_variant[:3]:
                    h    = env.root_states[ei_dbg.item(), 2].item()
                    h_lo = runner.base_height[ei_dbg.item()].item() * 0.5
                    h_hi = runner.base_height[ei_dbg.item()].item() * 1.5
                    print(f"  step {step} env {ei_dbg.item()}: "
                          f"h={h:.3f}  h_lo={h_lo:.3f}  h_hi={h_hi:.3f}  "
                          f"term={h < h_lo or h > h_hi}")

            MIN_EP_STEPS = 5
            for ei in env_ids_variant:
                ei = ei.item()
                if dones[ei].item() or ep_step_counters[ei].item() >= max_ep_steps:
                    ep_len = ep_step_counters[ei].item()
                    ep_step_counters[ei] = 0
                    if ep_len < MIN_EP_STEPS:
                        continue
                    episode_lengths.append(ep_len)
                    collected += 1
                    if collected % 5 == 0 or collected <= 3:
                        print(f"  Episode {collected}: len={ep_len}  "
                              f"({collected}/{num_rollouts * n_variant_envs})")
                    if collected >= num_rollouts * n_variant_envs:
                        break

            if step > max_ep_steps * num_rollouts * 3:
                print("  [Warning] Safety exit")
                break

    return episode_lengths


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
    if args.variant_name not in variant_names:
        raise ValueError(f"'{args.variant_name}' not in metadata. "
                         f"Available: {variant_names}")
    variant_idx = variant_names.index(args.variant_name)
    print(f"Variant '{args.variant_name}' (idx={variant_idx})")
    print(f"Mode: {'BASELINE' if args.baseline else 'FILM+RWSE+GCN'}")
    print(f"OOD: {args.ood}")

    cfg.PPO.NUM_ENVS     = args.num_envs
    cfg.MODEL.MAX_LIMBS  = 13
    cfg.MODEL.MAX_JOINTS = 12

    ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
    has_film  = any("film_generator" in k for k in ckpt_peek["model_state_dict"])
    has_gcn   = any("gcn.layers" in k     for k in ckpt_peek["model_state_dict"])
    del ckpt_peek

    if args.baseline:
        cfg.MODEL.GRAPH_ENCODING       = "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = False
    else:
        cfg.MODEL.GRAPH_ENCODING       = "rwse" if has_gcn else "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = has_film
        if has_gcn:
            cfg.MODEL.RWSE_K         = 8
            cfg.MODEL.GCN.HIDDEN_DIM = 16
            cfg.MODEL.GCN.OUT_DIM    = 13
            cfg.MODEL.GCN.NUM_LAYERS = 4
        print(f"  Auto-detected: film={has_film}  gcn={has_gcn}")

    cfg.DEVICE      = args.rl_device
    cfg.ENV.WALKERS = variant_names

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
    print(f"runner.base_height sample: {runner.base_height[:6].tolist()}")
    print(f"meta base_height_target: {meta[args.variant_name].get('base_height_target')}")

    # ── OOD mode: override obs stats with mean across training variants ───
    if args.ood:
        ob_mean_avg = runner.ob_mean.mean(0, keepdim=True).expand(
            runner.num_variants, -1).clone()
        ob_var_avg  = runner.ob_var.mean(0, keepdim=True).expand(
            runner.num_variants, -1).clone()
        runner.ob_mean = ob_mean_avg
        runner.ob_var  = ob_var_avg
        print(f"OOD mode: using mean obs stats for all {runner.num_variants} variants")
    else:
        print(f"In-distribution mode: using per-variant obs stats")

    runner.commands[:, 0] = args.cmd_vx
    runner.commands[:, 1] = args.cmd_vy
    runner.commands[:, 2] = args.cmd_yaw

    episode_lengths = run_eval_rollouts(
        runner, env, variant_idx,
        num_rollouts  = args.num_eval_rollouts,
        max_ep_steps  = args.max_ep_steps,
        device        = torch.device(args.rl_device),
        ood_mode      = args.ood,
    )

    avg_ep_len = float(np.mean(episode_lengths))
    std_ep_len = float(np.std(episode_lengths))
    med_ep_len = float(np.median(episode_lengths))
    min_ep_len = int(np.min(episode_lengths))
    max_ep_len = int(np.max(episode_lengths))

    print(f"\n── Eval Results for '{args.variant_name}' ──")
    print(f"  Model    : {'Baseline' if args.baseline else 'FiLM+RWSE+GCN'}")
    print(f"  OOD mode : {args.ood}")
    print(f"  Episodes : {len(episode_lengths)}")
    print(f"  Avg len  : {avg_ep_len:.1f} ± {std_ep_len:.1f} steps")
    print(f"  Median   : {med_ep_len:.1f} steps")
    print(f"  Min/Max  : {min_ep_len} / {max_ep_len} steps")

    # ── Record trajectory ─────────────────────────────────────────────────
    print(f"\n── Recording {args.num_steps} steps ──")
    env.reset_idx(torch.arange(env.num_envs, device=torch.device(args.rl_device)))
    runner.commands[:, 0] = args.cmd_vx
    obs = runner._get_obs_normalized()

    env_ids_variant = (env.env_variant_ids == variant_idx).nonzero(
        as_tuple=True)[0]
    ei = env_ids_variant[0].item()
    print(f"Recording env {ei}")

    traj = []
    runner.actor_critic.eval()
    with torch.no_grad():
        for step in range(args.num_steps):
            val, act, logp, dmv, dmmu = runner.agent.act(
                obs, unimal_ids=[0] * env.num_envs)
            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)
            obs, _, _, _ = runner._step(real_actions)
            obs = runner._normalize_obs(obs, update_stats=False)

            root = env.root_states[ei].cpu().numpy()
            dof  = env.dof_pos[ei].cpu().numpy()
            traj.append({
                "xyz":     root[:3].copy(),
                "quat":    root[3:7].copy(),
                "dof_pos": dof.copy(),
                "step":    step,
            })
            if step % 100 == 0:
                print(f"  step {step}/{args.num_steps}  height={root[2]:.3f}")

    output = {
        "trajectory":    traj,
        "variant_name":  args.variant_name,
        "variant_idx":   variant_idx,
        "cmd":           [args.cmd_vx, args.cmd_vy, args.cmd_yaw],
        "num_steps":     args.num_steps,
        "checkpoint":    args.checkpoint,
        "dof_names":     env.dof_names,
        "is_baseline":   args.baseline,
        "ood":           args.ood,
        "eval": {
            "episode_lengths": episode_lengths,
            "avg_ep_len":      avg_ep_len,
            "std_ep_len":      std_ep_len,
            "med_ep_len":      med_ep_len,
            "min_ep_len":      min_ep_len,
            "max_ep_len":      max_ep_len,
            "num_rollouts":    len(episode_lengths),
            "max_ep_steps":    args.max_ep_steps,
        },
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)

    print(f"\n── Summary ──")
    print(f"  Variant  : {args.variant_name}")
    print(f"  Model    : {'Baseline' if args.baseline else 'FiLM+RWSE+GCN'}")
    print(f"  Avg ep   : {avg_ep_len:.1f} ± {std_ep_len:.1f} steps")
    print(f"  Saved    : {args.out}  ({len(traj)} frames)")


if __name__ == "__main__":
    main()