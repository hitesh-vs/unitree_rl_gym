"""
record_best_traj.py

Records every episode for the tracked env to a folder of pkl files.
Each pkl is one episode — a few KB each.
At the end prints which pkl had the best ep_len.

Usage:
    python record_best_traj.py \
        --checkpoint output_film_wide/.../model_400.pt \
        --xml_path ... --variants_metadata ... \
        --variant_name armature_perturbed_robot_1 \
        --init_search_rollouts 850 \
        --out_dir trajs/film_armature1
"""

import isaacgym
from isaacgym import gymapi

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
    p.add_argument("--checkpoint",           required=True)
    p.add_argument("--xml_path",             required=True)
    p.add_argument("--variants_metadata",    required=True)
    p.add_argument("--variant_name",         required=True)
    p.add_argument("--baseline",             action="store_true", default=False)
    p.add_argument("--ood",                  action="store_true", default=False)
    p.add_argument("--init_search_rollouts", type=int,   default=850)
    p.add_argument("--min_ep_filter",        type=int,   default=10)
    p.add_argument("--max_ep_steps",         type=int,   default=1000)
    p.add_argument("--num_envs",             type=int,   default=512)
    p.add_argument("--track_env_offset",     type=int,   default=20)
    p.add_argument("--cmd_vx",               type=float, default=0.5)
    p.add_argument("--cmd_vy",               type=float, default=0.0)
    p.add_argument("--cmd_yaw",              type=float, default=0.0)
    p.add_argument("--sim_device",           type=str,   default="cuda:0")
    p.add_argument("--rl_device",            type=str,   default="cuda:0")
    p.add_argument("--out_dir",              type=str,   default="trajs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from modular_policy.config import cfg
    from modular_policy.algos.ppo.runner import ModularRunner
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
    from legged_gym.utils.helpers import parse_sim_params, class_to_dict, set_seed

    with open(args.variants_metadata) as f:
        meta = json.load(f)
    variant_names = list(meta.keys())
    if args.variant_name not in variant_names:
        raise ValueError(f"'{args.variant_name}' not in metadata.")
    variant_idx = variant_names.index(args.variant_name)
    print(f"Variant '{args.variant_name}' (idx={variant_idx})")
    print(f"Saving all episodes → {args.out_dir}/")

    cfg.PPO.NUM_ENVS     = args.num_envs
    cfg.MODEL.MAX_LIMBS  = 13
    cfg.MODEL.MAX_JOINTS = 12

    ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
    has_film  = any("film_generator" in k for k in ckpt_peek["model_state_dict"])
    has_gcn   = any("gcn.layers"     in k for k in ckpt_peek["model_state_dict"])
    del ckpt_peek

    if args.baseline:
        cfg.MODEL.GRAPH_ENCODING       = "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = False
    else:
        cfg.MODEL.GRAPH_ENCODING       = "rwse" if has_gcn else "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = has_film
        if has_gcn:
            cfg.MODEL.RWSE_K=8; cfg.MODEL.GCN.HIDDEN_DIM=16
            cfg.MODEL.GCN.OUT_DIM=13; cfg.MODEL.GCN.NUM_LAYERS=4
        print(f"  film={has_film}  gcn={has_gcn}")

    cfg.DEVICE      = args.rl_device
    cfg.ENV.WALKERS = variant_names

    set_seed(1409)
    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs

    class _Args:
        physics_engine=gymapi.SIM_PHYSX; sim_device=args.sim_device
        rl_device=args.rl_device; headless=True; use_gpu=True
        use_gpu_pipeline=True; subscenes=0; num_threads=10
        num_envs=None; seed=None; max_iterations=None; resume=False
        experiment_name=None; run_name=None; load_run=None; checkpoint=None
        device=args.rl_device; sim_device_type="cuda"
        compute_device_id=0; num_subscenes=0

    sim_params = parse_sim_params(_Args(), {"sim": class_to_dict(env_cfg.sim)})
    env = MultiVariantG1Robot(
        cfg=env_cfg, sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=args.sim_device, headless=True,
        variants_metadata_path=args.variants_metadata)
    print(f"Env ready — {env.num_envs} envs")

    log_dir = f"/tmp/record_{datetime.now().strftime('%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    runner = ModularRunner(
        env=env, xml_path=os.path.abspath(args.xml_path),
        log_dir=log_dir, device=args.rl_device,
        variants_metadata_path=args.variants_metadata)
    runner.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")

    if args.ood:
        if runner.ob_mean.dim() == 2:
            runner.ob_mean = runner.ob_mean.mean(0)
            runner.ob_var  = runner.ob_var.mean(0)
        print("OOD: mean obs stats")

    runner.commands[:, 0] = args.cmd_vx
    runner.commands[:, 1] = args.cmd_vy
    runner.commands[:, 2] = args.cmd_yaw

    device    = torch.device(args.rl_device)
    fi        = runner.obs_builder.feet_indices
    env_ids_v = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
    n_envs    = len(env_ids_v)
    track_ei  = env_ids_v[min(args.track_env_offset, n_envs-1)].item()

    print(f"Tracking env {track_ei}  ({args.init_search_rollouts} episodes)\n")

    ep_counters       = torch.zeros(env.num_envs, dtype=torch.long,    device=device)
    step_counts       = torch.zeros(env.num_envs, dtype=torch.long,    device=device)
    foot_contact_prev = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=device)

    live_buf    = []
    live_phys   = 0
    live_prev_c = torch.zeros(2, dtype=torch.bool, device=device)
    ep_num      = 0
    collected   = 0
    step        = 0
    all_ep_lens = []

    env.reset_idx(torch.arange(env.num_envs, device=device))
    runner.commands[:, 0] = 0.5
    obs = runner._get_obs_normalized(update_stats=False)
    runner.actor_critic.eval()

    with torch.no_grad():
        while collected < args.init_search_rollouts:
            _, act, _, _, _ = runner.agent.act(obs, unimal_ids=[0]*env.num_envs)
            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)
            obs, _, dones, _ = runner._step(real_actions)
            obs = runner._normalize_obs(obs)
            ep_counters += 1
            step        += 1

            # Physical step counter
            lf_z = env.contact_forces[:env.num_envs, fi[0], 2].abs() > 1.
            rf_z = env.contact_forces[:env.num_envs, fi[1], 2].abs() > 1.
            contact_now  = torch.stack([lf_z, rf_z], dim=1)
            touchdown    = (~foot_contact_prev) & contact_now
            step_counts += touchdown.any(dim=1).long()
            foot_contact_prev = contact_now.clone()

            # Live frame for tracked env
            root   = env.root_states[track_ei].cpu().numpy()
            dof    = env.dof_pos[track_ei].cpu().numpy()
            lf_z_v = env.contact_forces[track_ei, fi[0], 2].item()
            rf_z_v = env.contact_forces[track_ei, fi[1], 2].item()
            lf_c   = abs(lf_z_v) > 1.
            rf_c   = abs(rf_z_v) > 1.
            cur_c  = torch.tensor([lf_c, rf_c], dtype=torch.bool, device=device)
            if ((~live_prev_c) & cur_c).any():
                live_phys += 1
            live_prev_c = cur_c.clone()
            live_buf.append({
                "xyz":       root[:3].copy(),
                "quat":      root[3:7].copy(),
                "dof_pos":   dof.copy(),
                "step":      len(live_buf),
                "left_fz":   lf_z_v,
                "right_fz":  rf_z_v,
                "walking":   lf_c or rf_c,
                "phys_step": live_phys,
            })

            # Check all variant envs
            for ei in env_ids_v:
                ei = ei.item()
                if dones[ei].item() or ep_counters[ei].item() >= args.max_ep_steps:
                    ep_len     = ep_counters[ei].item()
                    phys_steps = step_counts[ei].item()
                    ep_counters[ei] = 0
                    step_counts[ei] = 0
                    foot_contact_prev[ei] = False

                    if ei == track_ei and ep_len >= args.min_ep_filter:
                        ep_num    += 1
                        collected += 1
                        walk_pct   = (sum(1 for f in live_buf if f["walking"])
                                      / max(len(live_buf), 1) * 100)
                        all_ep_lens.append(ep_len)

                        pkl_path = os.path.join(
                            args.out_dir, f"ep_{ep_num:04d}_len{ep_len}_steps{phys_steps}.pkl")
                        with open(pkl_path, "wb") as f:
                            pickle.dump({
                                "trajectory":      live_buf,
                                "dof_names":       env.dof_names,
                                "variant_name":    args.variant_name,
                                "ep_num":          ep_num,
                                "ep_len":          ep_len,
                                "physical_steps":  phys_steps,
                                "walk_quality_pct": walk_pct,
                            }, f)

                        print(f"  ep {ep_num:3d}: len={ep_len:4d}  "
                              f"phys={phys_steps:3d}  "
                              f"walk={walk_pct:.0f}%  "
                              f"→ {os.path.basename(pkl_path)}")

                        # Reset live buffer
                        live_buf    = []
                        live_phys   = 0
                        live_prev_c = torch.zeros(2, dtype=torch.bool, device=device)

            if step > args.max_ep_steps * args.init_search_rollouts * 3:
                print("  [Warning] Safety exit")
                break

    if not all_ep_lens:
        print("No valid episodes recorded.")
        return

    best_idx = int(np.argmax(all_ep_lens))
    print(f"\n── Summary ──")
    print(f"  Episodes recorded : {ep_num}")
    print(f"  Avg ep_len        : {np.mean(all_ep_lens):.1f} ± {np.std(all_ep_lens):.1f}")
    print(f"  Best ep_len       : {max(all_ep_lens)}  (ep {best_idx+1})")
    print(f"  All saved to      : {args.out_dir}/")
    print(f"\n  Best pkl: ep_{best_idx+1:04d}_len{max(all_ep_lens)}_*.pkl")


if __name__ == "__main__":
    main()