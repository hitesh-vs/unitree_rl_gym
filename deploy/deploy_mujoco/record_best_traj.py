"""
record_traj_isaac.py

Two-phase evaluation:
  Phase 1 (--find_best_init): Run many rollouts, save the initial state
                               that gave the longest episode to a .pt file.
                               Also counts physical steps (foot touchdowns).
  Phase 2 (--replay_init):    Load that saved init state, reproduce the
                               exact same episode deterministically.

Usage:
    # Phase 1 — find and save best init
    python record_traj_isaac.py \
        --checkpoint output_film_wide/.../model_400.pt \
        --xml_path ... --variants_metadata ... \
        --variant_name armature_perturbed_robot_1 \
        --find_best_init \
        --init_search_rollouts 200 \
        --init_save best_init_armature1.pt \
        --out search_results.pkl

    # Phase 2 — replay best init exactly
    python record_traj_isaac.py \
        --checkpoint output_film_wide/.../model_400.pt \
        --xml_path ... --variants_metadata ... \
        --variant_name armature_perturbed_robot_1 \
        --replay_init best_init_armature1.pt \
        --num_eval_rollouts 10 \
        --out traj_film_armature1.pkl
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
    p.add_argument("--find_best_init",       action="store_true", default=False)
    p.add_argument("--init_search_rollouts", type=int, default=200)
    p.add_argument("--init_save",            type=str, default="best_init.pt")
    p.add_argument("--replay_init",          type=str, default=None)
    p.add_argument("--num_eval_rollouts",    type=int, default=10)
    p.add_argument("--min_ep_filter",        type=int, default=10)
    p.add_argument("--num_steps",            type=int,   default=1000)
    p.add_argument("--max_ep_steps",         type=int,   default=1000)
    p.add_argument("--num_envs",             type=int,   default=512)
    p.add_argument("--cmd_vx",               type=float, default=0.5)
    p.add_argument("--cmd_vy",               type=float, default=0.0)
    p.add_argument("--cmd_yaw",              type=float, default=0.0)
    p.add_argument("--sim_device",           type=str,   default="cuda:0")
    p.add_argument("--rl_device",            type=str,   default="cuda:0")
    p.add_argument("--out",                  type=str,   default="trajectory.pkl")
    return p.parse_args()


def apply_init(env, env_ids, init_state, device):
    from isaacgym import gymtorch
    for ei in env_ids:
        ei = ei.item() if hasattr(ei, 'item') else int(ei)
        env.root_states[ei] = init_state["root"].to(device)
        env.dof_pos[ei]     = init_state["dof_pos"].to(device)
        env.dof_vel[ei]     = init_state["dof_vel"].to(device)
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.set_dof_state_tensor(
        env.sim, gymtorch.unwrap_tensor(
            torch.cat([env.dof_pos, env.dof_vel], dim=-1)))


# ── Phase 1: search ───────────────────────────────────────────────────────────

def phase_find_best_init(runner, env, variant_idx, n_search,
                          max_ep_steps, min_ep_filter, device, save_path):
    env_ids_v = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
    n_envs    = len(env_ids_v)
    print(f"\n[Phase 1] Searching {n_search} episodes across {n_envs} envs "
          f"of variant {variant_idx}...")
    print(f"  Will save best init → {save_path}")

    ep_counters       = torch.zeros(env.num_envs, dtype=torch.long,  device=device)
    step_counts       = torch.zeros(env.num_envs, dtype=torch.long,  device=device)
    foot_contact_prev = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=device)

    best_ep_len  = -1
    best_steps   = 0
    best_init    = None
    collected    = 0
    step         = 0

    # Get foot indices
    fi = runner.obs_builder.feet_indices   # [left_idx, right_idx]

    env.reset_idx(torch.arange(env.num_envs, device=device))
    env_init_snap = {}
    for ei in env_ids_v:
        ei = ei.item()
        env_init_snap[ei] = {
            "root":    env.root_states[ei].clone().cpu(),
            "dof_pos": env.dof_pos[ei].clone().cpu(),
            "dof_vel": env.dof_vel[ei].clone().cpu(),
        }

    runner.commands[:, 0] = 0.5
    obs = runner._get_obs_normalized(update_stats=False)
    runner.actor_critic.eval()

    with torch.no_grad():
        while collected < n_search:
            _, act, _, _, _ = runner.agent.act(obs, unimal_ids=[0]*env.num_envs)
            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)
            obs, _, dones, _ = runner._step(real_actions)
            obs = runner._normalize_obs(obs)
            ep_counters += 1
            step        += 1

            # ── Count physical steps via foot touchdowns ──────────────────
            lf_z = env.contact_forces[:env.num_envs, fi[0], 2].abs() > 1.
            rf_z = env.contact_forces[:env.num_envs, fi[1], 2].abs() > 1.
            contact_now  = torch.stack([lf_z, rf_z], dim=1)  # (N, 2)
            touchdown    = (~foot_contact_prev) & contact_now
            step_counts += touchdown.any(dim=1).long()
            foot_contact_prev = contact_now.clone()

            for ei in env_ids_v:
                ei = ei.item()
                if dones[ei].item() or ep_counters[ei].item() >= max_ep_steps:
                    ep_len   = ep_counters[ei].item()
                    phys_steps = step_counts[ei].item()
                    ep_counters[ei] = 0
                    step_counts[ei] = 0
                    foot_contact_prev[ei] = False

                    if ep_len >= min_ep_filter:
                        collected += 1
                        if ep_len > best_ep_len:
                            best_ep_len = ep_len
                            best_steps  = phys_steps
                            best_init   = {k: v.clone()
                                           for k, v in env_init_snap[ei].items()}
                            print(f"  New best: env {ei}  "
                                  f"ep_len={ep_len}  "
                                  f"physical_steps={phys_steps}  "
                                  f"({collected}/{n_search})")

                    # Reset and snapshot new init for this env
                    env.reset_idx(torch.tensor([ei], device=device))
                    env_init_snap[ei] = {
                        "root":    env.root_states[ei].clone().cpu(),
                        "dof_pos": env.dof_pos[ei].clone().cpu(),
                        "dof_vel": env.dof_vel[ei].clone().cpu(),
                    }

            if step > max_ep_steps * n_search * 3:
                print("  [Warning] Safety exit")
                break

    if best_init is None:
        print("  No valid episode found!")
        return None, -1, 0

    torch.save({
        "init_state":     best_init,
        "best_ep_len":    best_ep_len,
        "best_steps":     best_steps,
        "variant_name":   variant_idx,
        "search_n":       n_search,
    }, save_path)

    print(f"\n[Phase 1 done] Best ep_len={best_ep_len}  "
          f"physical_steps={best_steps}  saved → {save_path}")
    return best_init, best_ep_len, best_steps


# ── Phase 2: replay ───────────────────────────────────────────────────────────

def phase_replay_init(runner, env, variant_idx, init_state,
                       num_rollouts, max_ep_steps, device):
    env_ids_v = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
    print(f"\n[Phase 2] Replaying saved init across {len(env_ids_v)} envs, "
          f"{num_rollouts} episodes...")

    ep_counters       = torch.zeros(env.num_envs, dtype=torch.long,   device=device)
    step_counts       = torch.zeros(env.num_envs, dtype=torch.long,   device=device)
    foot_contact_prev = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=device)
    fi                = runner.obs_builder.feet_indices

    episode_lengths = []
    physical_steps  = []
    collected       = 0
    step            = 0

    env.reset_idx(torch.arange(env.num_envs, device=device))
    apply_init(env, env_ids_v, init_state, device)

    runner.commands[:, 0] = 0.5
    obs = runner._get_obs_normalized(update_stats=False)
    runner.actor_critic.eval()

    with torch.no_grad():
        while collected < num_rollouts:
            _, act, _, _, _ = runner.agent.act(obs, unimal_ids=[0]*env.num_envs)
            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)
            obs, _, dones, _ = runner._step(real_actions)
            obs = runner._normalize_obs(obs)
            ep_counters += 1
            step        += 1

            # Count physical steps
            lf_z = env.contact_forces[:env.num_envs, fi[0], 2].abs() > 1.
            rf_z = env.contact_forces[:env.num_envs, fi[1], 2].abs() > 1.
            contact_now  = torch.stack([lf_z, rf_z], dim=1)
            touchdown    = (~foot_contact_prev) & contact_now
            step_counts += touchdown.any(dim=1).long()
            foot_contact_prev = contact_now.clone()

            for ei in env_ids_v:
                ei = ei.item()
                if dones[ei].item() or ep_counters[ei].item() >= max_ep_steps:
                    ep_len     = ep_counters[ei].item()
                    phys_steps = step_counts[ei].item()
                    ep_counters[ei] = 0
                    step_counts[ei] = 0
                    foot_contact_prev[ei] = False

                    env.reset_idx(torch.tensor([ei], device=device))
                    apply_init(env, [ei], init_state, device)

                    if ep_len < 5:
                        continue
                    episode_lengths.append(ep_len)
                    physical_steps.append(phys_steps)
                    collected += 1
                    print(f"  Replay ep {collected}: "
                          f"ep_len={ep_len}  physical_steps={phys_steps}  "
                          f"({collected}/{num_rollouts})")
                    if collected >= num_rollouts:
                        break

            if step > max_ep_steps * num_rollouts * 3:
                print("  [Warning] Safety exit")
                break

    avg     = float(np.mean(episode_lengths))
    std     = float(np.std(episode_lengths))
    avg_ps  = float(np.mean(physical_steps))
    print(f"\n[Phase 2 done] avg_ep={avg:.1f}±{std:.1f}  "
          f"avg_physical_steps={avg_ps:.1f}  "
          f"min/max={min(episode_lengths)}/{max(episode_lengths)}")
    return episode_lengths, physical_steps


# ── Main ──────────────────────────────────────────────────────────────────────

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
        raise ValueError(f"'{args.variant_name}' not in metadata.")
    variant_idx = variant_names.index(args.variant_name)
    print(f"Variant '{args.variant_name}' (idx={variant_idx})")

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

    device = torch.device(args.rl_device)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    if args.find_best_init:
        best_init, best_ep_len, best_steps = phase_find_best_init(
            runner, env, variant_idx,
            n_search      = args.init_search_rollouts,
            max_ep_steps  = args.max_ep_steps,
            min_ep_filter = args.min_ep_filter,
            device        = device,
            save_path     = args.init_save)

        if best_init is None:
            print("Search found nothing. Exiting.")
            return

        eps, phys = phase_replay_init(runner, env, variant_idx,
                                       best_init, 5, args.max_ep_steps, device)
        output = {
            "mode":           "find_best_init",
            "best_ep_len":    best_ep_len,
            "best_steps":     best_steps,
            "init_save":      args.init_save,
            "confirm_replay_eps":   eps,
            "confirm_replay_steps": phys,
        }
        with open(args.out, "wb") as f:
            pickle.dump(output, f)
        print(f"Saved search results → {args.out}")
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────
    if args.replay_init:
        saved       = torch.load(args.replay_init, map_location="cpu")
        init_state  = saved["init_state"]
        orig_ep_len = saved.get("best_ep_len", "unknown")
        orig_steps  = saved.get("best_steps",  "unknown")
        print(f"\nLoaded init → {args.replay_init}  "
              f"(originally: ep_len={orig_ep_len}  physical_steps={orig_steps})")

        eps, phys = phase_replay_init(runner, env, variant_idx,
                                       init_state, args.num_eval_rollouts,
                                       args.max_ep_steps, device)

        avg_ep = float(np.mean(eps))
        std_ep = float(np.std(eps))
        avg_ps = float(np.mean(phys))
        std_ps = float(np.std(phys))

        print(f"\n── Replay Results for '{args.variant_name}' ──")
        print(f"  Original        : ep_len={orig_ep_len}  physical_steps={orig_steps}")
        print(f"  Replayed avg    : ep_len={avg_ep:.1f}±{std_ep:.1f}  "
              f"physical_steps={avg_ps:.1f}±{std_ps:.1f}")
        print(f"  Min/Max ep      : {min(eps)}/{max(eps)}")
        print(f"  Min/Max steps   : {min(phys)}/{max(phys)}")

        # Record trajectory
        print(f"\n── Recording {args.num_steps} steps ──")
        env.reset_idx(torch.arange(env.num_envs, device=device))
        env_ids_v = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
        apply_init(env, env_ids_v, init_state, device)
        runner.commands[:, 0] = args.cmd_vx
        obs = runner._get_obs_normalized(update_stats=False)
        ei  = env_ids_v[0].item()

        fi             = runner.obs_builder.feet_indices
        traj           = []
        phys_step_cnt  = 0
        prev_contact   = torch.zeros(2, dtype=torch.bool, device=device)

        runner.actor_critic.eval()
        with torch.no_grad():
            for step in range(args.num_steps):
                _, act, _, _, _ = runner.agent.act(obs, unimal_ids=[0]*env.num_envs)
                act_mask     = runner.obs_builder.act_padding_mask[0].bool()
                real_actions = act[:, ~act_mask].clamp(-1., 1.)
                obs, _, _, _ = runner._step(real_actions)
                obs          = runner._normalize_obs(obs, update_stats=False)

                root = env.root_states[ei].cpu().numpy()
                dof  = env.dof_pos[ei].cpu().numpy()

                lf_z = env.contact_forces[ei, fi[0], 2].item()
                rf_z = env.contact_forces[ei, fi[1], 2].item()
                lf_c = abs(lf_z) > 1.
                rf_c = abs(rf_z) > 1.

                # Count touchdown events
                cur_contact = torch.tensor([lf_c, rf_c], dtype=torch.bool, device=device)
                if ((~prev_contact) & cur_contact).any():
                    phys_step_cnt += 1
                prev_contact = cur_contact.clone()

                traj.append({
                    "xyz":          root[:3].copy(),
                    "quat":         root[3:7].copy(),
                    "dof_pos":      dof.copy(),
                    "step":         step,
                    "left_fz":      lf_z,
                    "right_fz":     rf_z,
                    "walking":      lf_c or rf_c,
                    "phys_step":    phys_step_cnt,
                })
                if step % 100 == 0:
                    print(f"  step {step}/{args.num_steps}  "
                          f"h={root[2]:.3f}  "
                          f"physical_steps_so_far={phys_step_cnt}")

        walk_pct = sum(1 for f in traj if f["walking"]) / len(traj) * 100
        print(f"\nTotal physical steps in recorded trajectory: {phys_step_cnt}")
        print(f"Walk quality: {walk_pct:.1f}% steps with foot contact")

        output = {
            "mode":                  "replay",
            "trajectory":            traj,
            "dof_names":             env.dof_names,
            "variant_name":          args.variant_name,
            "original_best_ep_len":  orig_ep_len,
            "original_best_steps":   orig_steps,
            "replayed_eps":          eps,
            "replayed_phys_steps":   phys,
            "avg_ep":                avg_ep,
            "std_ep":                std_ep,
            "avg_phys_steps":        avg_ps,
            "std_phys_steps":        std_ps,
            "traj_phys_steps":       phys_step_cnt,
            "walk_quality_pct":      walk_pct,
        }
        with open(args.out, "wb") as f:
            pickle.dump(output, f)
        print(f"Saved → {args.out}  ({len(traj)} frames)")
        return

    print("Specify --find_best_init or --replay_init")


if __name__ == "__main__":
    main()