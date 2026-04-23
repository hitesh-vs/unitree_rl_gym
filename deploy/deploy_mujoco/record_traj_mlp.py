"""
record_traj_isaac.py

Unified eval script for transformer (FiLM / baseline) and flat MLP checkpoints.
Auto-detects checkpoint type. Same interface for all three models.

Usage:
    # FiLM or baseline transformer (auto detected)
    python record_traj_isaac.py \
        --checkpoint output_film_wide/Mar31_18-18-17/model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --variant_name robot_variant_4 \
        --num_eval_rollouts 10 \
        --out traj_film_v4.pkl

    # Flat MLP (same command, auto detected from checkpoint)
    python record_traj_isaac.py \
        --checkpoint output_flat_mlp/.../model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --variant_name robot_variant_4 \
        --num_eval_rollouts 10 \
        --out traj_mlp_v4.pkl

    # Force baseline transformer
    python record_traj_isaac.py --baseline ...

    # OOD mode (mean obs stats)
    python record_traj_isaac.py --ood ...
"""

import isaacgym
from isaacgym import gymapi, gymtorch

import argparse
import os
import sys
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GAIT_PERIOD = 0.8
GAIT_OFFSET = 0.5


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--xml_path",          required=True)
    p.add_argument("--variants_metadata", required=True)
    p.add_argument("--variant_name",      required=True)
    p.add_argument("--baseline",          action="store_true", default=False)
    p.add_argument("--ood",               action="store_true", default=False)
    p.add_argument("--num_steps",         type=int,   default=500)
    p.add_argument("--num_eval_rollouts", type=int,   default=10)
    p.add_argument("--max_ep_steps",      type=int,   default=1000)
    p.add_argument("--num_envs",          type=int,   default=512)
    p.add_argument("--cmd_vx",            type=float, default=0.5)
    p.add_argument("--cmd_vy",            type=float, default=0.0)
    p.add_argument("--cmd_yaw",           type=float, default=0.0)
    p.add_argument("--sim_device",        type=str,   default="cuda:0")
    p.add_argument("--rl_device",         type=str,   default="cuda:0")
    p.add_argument("--out",               type=str,   default="trajectory.pkl")
    return p.parse_args()


# ── Shared env builder ────────────────────────────────────────────────────────

def build_env(args, env_cfg):
    from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
    from legged_gym.utils.helpers import parse_sim_params, class_to_dict

    class _Args:
        physics_engine    = gymapi.SIM_PHYSX
        sim_device        = args.sim_device
        rl_device         = args.rl_device
        headless          = True
        use_gpu           = True
        use_gpu_pipeline  = True
        subscenes         = 0
        num_threads       = 10
        num_envs          = None; seed = None; max_iterations = None
        resume            = False; experiment_name = None; run_name = None
        load_run          = None; checkpoint = None
        device            = args.rl_device
        sim_device_type   = "cuda"
        compute_device_id = 0
        num_subscenes     = 0

    sim_params = parse_sim_params(_Args(), {"sim": class_to_dict(env_cfg.sim)})
    return MultiVariantG1Robot(
        cfg                    = env_cfg,
        sim_params             = sim_params,
        physics_engine         = gymapi.SIM_PHYSX,
        sim_device             = args.sim_device,
        headless               = True,
        variants_metadata_path = args.variants_metadata,
    )


# ── Generic eval loop ─────────────────────────────────────────────────────────

def eval_rollouts(step_fn, reset_fn, env, variant_idx,
                  num_rollouts, max_ep_steps, device):
    env_ids = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
    n_envs  = len(env_ids)
    target  = num_rollouts * n_envs
    print(f"\n[Eval] {num_rollouts} rollouts × {n_envs} envs = {target} episodes")

    ep_counters     = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    episode_lengths = []
    collected = 0
    step      = 0

    env.reset_idx(torch.arange(env.num_envs, device=device))
    obs = reset_fn()

    MIN_EP = 5
    while collected < target:
        obs, dones = step_fn(obs)
        ep_counters += 1
        step        += 1

        if step < 5:
            for ei_dbg in env_ids[:3]:
                h    = env.root_states[ei_dbg.item(), 2].item()
                print(f"  step {step} env {ei_dbg.item()}: h={h:.3f}")

        for ei in env_ids:
            ei = ei.item()
            if dones[ei].item() or ep_counters[ei].item() >= max_ep_steps:
                ep_len = ep_counters[ei].item()
                ep_counters[ei] = 0
                if ep_len < MIN_EP:
                    continue
                episode_lengths.append(ep_len)
                collected += 1
                if collected % 5 == 0 or collected <= 3:
                    print(f"  Episode {collected}: len={ep_len}  "
                          f"({collected}/{target})")
                if collected >= target:
                    break

        if step > max_ep_steps * num_rollouts * 3:
            print("  [Warning] Safety exit")
            break

    return episode_lengths


# ── MLP helpers ───────────────────────────────────────────────────────────────

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(256, 256, 128)):
        super().__init__()
        def make_mlp(out_dim):
            layers = []; in_dim = obs_dim
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.ELU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)
        self.actor   = make_mlp(action_dim)
        self.critic  = make_mlp(1)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, obs):
        return self.actor(obs)   # deterministic at eval


def build_flat_obs(env, commands, episode_steps, last_actions,
                   joint_lo, joint_span, dt, device):
    N = env.num_envs
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    root     = env.root_states[:N]
    dof_pos  = env.dof_pos[:N]
    dof_vel  = env.dof_vel[:N]
    q_wxyz   = torch.cat([root[:,6:7], root[:,3:6]], dim=-1)

    def qrot_inv(q, v):
        w=q[...,0:1]; xyz=q[...,1:]
        t=2.*torch.cross(xyz,v,dim=-1)
        return v - w*t + torch.cross(xyz,t,dim=-1)

    ang_vel  = qrot_inv(q_wxyz, root[:,10:13]) * 0.25
    grav_w   = torch.tensor([0.,0.,-1.], device=device).expand(N,3)
    proj_g   = qrot_inv(q_wxyz, grav_w)
    cmd_s    = commands * torch.tensor([2.,2.,.25], device=device)
    t        = episode_steps.float() * dt
    pl       = (t % GAIT_PERIOD) / GAIT_PERIOD
    pr       = (pl + GAIT_OFFSET) % 1.
    sin_ph   = torch.sin(2.*math.pi*pl).unsqueeze(1)
    cos_ph   = torch.cos(2.*math.pi*pl).unsqueeze(1)
    dpn      = (dof_pos - joint_lo) / joint_span * 2 - 1
    dv       = dof_vel.clamp(-10,10)
    return torch.cat([ang_vel, proj_g, cmd_s, sin_ph, cos_ph, dpn, dv], dim=1), pl, pr


def termination_mlp(env, episode_steps, base_height):
    N  = env.num_envs
    r  = env.root_states[:N]
    h  = r[:,2]
    q  = torch.cat([r[:,6:7],r[:,3:6]],dim=-1)
    w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
    roll  = torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1,1))
    pf    = env.contact_forces[:N,0,2].abs()
    return ((pf>1.)|(h<base_height*0.5)|(h>base_height*1.5)|
            (pitch.abs()>1.)|(roll.abs()>.8)|(episode_steps>=1000))


# ── MLP eval ──────────────────────────────────────────────────────────────────

def run_mlp(args, meta, variant_names, variant_idx, ckpt):
    from modular_policy.config import cfg
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.utils.helpers import set_seed

    print("\n[Mode] Flat MLP")
    cfg.PPO.NUM_ENVS = args.num_envs
    cfg.ENV.WALKERS  = variant_names
    set_seed(1409)

    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs
    env                  = build_env(args, env_cfg)
    device               = torch.device(args.rl_device)
    print(f"Env ready — {env.num_envs} envs")

    model = MLPActorCritic(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ob_mean = ckpt["ob_mean"].to(device)
    ob_var  = ckpt["ob_var"].to(device)
    print(f"MLP: obs_dim={ckpt['obs_dim']}  action_dim={ckpt['action_dim']}")

    n  = env.num_actions
    dp = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    joint_lo   = torch.tensor([float(dp["lower"][i]) for i in range(n)],
                               dtype=torch.float32, device=device)
    joint_hi   = torch.tensor([float(dp["upper"][i]) for i in range(n)],
                               dtype=torch.float32, device=device)
    joint_span = (joint_hi - joint_lo).clamp(min=1e-8)

    bht = torch.tensor([m["base_height_target"] for m in meta.values()],
                        dtype=torch.float32, device=device)
    base_height = bht[env.env_variant_ids]

    dt            = env.cfg.control.decimation * env.sim_params.dt
    commands      = torch.zeros(env.num_envs, 3, device=device)
    commands[:,0] = args.cmd_vx
    commands[:,1] = args.cmd_vy
    commands[:,2] = args.cmd_yaw
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    last_actions  = torch.zeros(env.num_envs, n, device=device)

    def normalize(raw):
        return ((raw - ob_mean) / (ob_var + 1e-8).sqrt()).clamp(-10, 10)

    def reset_fn():
        raw, _, _ = build_flat_obs(env, commands, episode_steps,
                                    last_actions, joint_lo, joint_span, dt, device)
        return normalize(raw)

    def step_fn(obs):
        with torch.no_grad():
            real_actions = model.get_action(obs).clamp(-1., 1.)
        for _ in range(env.cfg.control.decimation):
            env.torques = env._compute_torques(real_actions)
            env.gym.set_dof_actuation_force_tensor(
                env.sim, gymtorch.unwrap_tensor(env.torques))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        episode_steps.add_(1)
        last_actions.copy_(real_actions)

        dones    = termination_mlp(env, episode_steps, base_height)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids):
            env.reset_idx(done_ids)
            episode_steps[done_ids] = 0
            last_actions[done_ids]  = 0

        raw, _, _ = build_flat_obs(env, commands, episode_steps,
                                    last_actions, joint_lo, joint_span, dt, device)
        return normalize(raw), dones

    eps = eval_rollouts(step_fn, reset_fn, env, variant_idx,
                        args.num_eval_rollouts, args.max_ep_steps, device)
    return eps, "Flat MLP", [], []


# ── Transformer eval ──────────────────────────────────────────────────────────

def run_transformer(args, meta, variant_names, variant_idx, ckpt):
    from modular_policy.config import cfg
    from modular_policy.algos.ppo.runner import ModularRunner
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.utils.helpers import set_seed

    has_film = any("film_generator" in k for k in ckpt["model_state_dict"])
    has_gcn  = any("gcn.layers"     in k for k in ckpt["model_state_dict"])

    if args.baseline:
        cfg.MODEL.GRAPH_ENCODING       = "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = False
        label = "Context + Transformer (baseline)"
        print("\n[Mode] Baseline transformer (forced)")
    else:
        cfg.MODEL.GRAPH_ENCODING       = "rwse" if has_gcn else "none"
        cfg.MODEL.TRANSFORMER.USE_FILM = has_film
        if has_gcn:
            cfg.MODEL.RWSE_K=8; cfg.MODEL.GCN.HIDDEN_DIM=16
            cfg.MODEL.GCN.OUT_DIM=13; cfg.MODEL.GCN.NUM_LAYERS=4
        label = ("Graph+RWSE+FiLM (ours)" if has_film
                 else "Context + Transformer (baseline)")
        print(f"\n[Mode] Transformer  film={has_film}  gcn={has_gcn}")

    cfg.PPO.NUM_ENVS     = args.num_envs
    cfg.MODEL.MAX_LIMBS  = 13
    cfg.MODEL.MAX_JOINTS = 12
    cfg.DEVICE           = args.rl_device
    cfg.ENV.WALKERS      = variant_names

    set_seed(1409)
    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs
    env                  = build_env(args, env_cfg)
    device               = torch.device(args.rl_device)
    print(f"Env ready — {env.num_envs} envs")

    log_dir = f"/tmp/record_{datetime.now().strftime('%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    runner = ModularRunner(
        env=env, xml_path=os.path.abspath(args.xml_path),
        log_dir=log_dir, device=args.rl_device,
        variants_metadata_path=args.variants_metadata)
    runner.load(args.checkpoint)
    print(f"Loaded: {args.checkpoint}")
    print(f"base_height sample: {runner.base_height[:4].tolist()}")
    print(f"meta base_height  : "
          f"{meta[args.variant_name].get('base_height_target')}")

    if args.ood:
        if runner.ob_mean.dim() == 1:
            print("OOD mode: ob_mean already 1D (shared), no change needed")
        else:
            runner.ob_mean = runner.ob_mean.mean(0)
            runner.ob_var  = runner.ob_var.mean(0)
        print("OOD mode: using mean obs stats")

    runner.commands[:,0] = args.cmd_vx
    runner.commands[:,1] = args.cmd_vy
    runner.commands[:,2] = args.cmd_yaw
    runner.actor_critic.eval()

    obs0 = runner._get_obs_normalized(update_stats=False)
    env_ids = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
    ctx = obs0["context"][env_ids[0].item()]
    print(f"\n[Context] mean={ctx.mean():.4f}  std={ctx.std():.4f}  "
          f"first4={ctx[:4].cpu().numpy().round(3)}")

    def reset_fn():
        return runner._get_obs_normalized(update_stats=False)

    def step_fn(obs):
        with torch.no_grad():
            _, act, _, _, _ = runner.agent.act(
                obs, unimal_ids=[0]*env.num_envs)
        act_mask     = runner.obs_builder.act_padding_mask[0].bool()
        real_actions = act[:, ~act_mask].clamp(-1., 1.)
        obs_next, _, dones, _ = runner._step(real_actions)
        obs_next = runner._normalize_obs(obs_next)
        return obs_next, dones

    eps = eval_rollouts(step_fn, reset_fn, env, variant_idx,
                        args.num_eval_rollouts, args.max_ep_steps, device)

    # Record trajectory
    print(f"\n── Recording {args.num_steps} steps ──")
    env.reset_idx(torch.arange(env.num_envs, device=device))
    runner.commands[:,0] = args.cmd_vx
    obs = runner._get_obs_normalized(update_stats=False)
    env_ids = (env.env_variant_ids == variant_idx).nonzero(as_tuple=True)[0]
    ei      = env_ids[0].item()

    traj = []
    with torch.no_grad():
        for step in range(args.num_steps):
            _, act, _, _, _ = runner.agent.act(obs, unimal_ids=[0]*env.num_envs)
            act_mask     = runner.obs_builder.act_padding_mask[0].bool()
            real_actions = act[:, ~act_mask].clamp(-1., 1.)
            obs, _, _, _ = runner._step(real_actions)
            obs          = runner._normalize_obs(obs, update_stats=False)
            root = env.root_states[ei].cpu().numpy()
            dof  = env.dof_pos[ei].cpu().numpy()
            traj.append({"xyz": root[:3].copy(), "quat": root[3:7].copy(),
                         "dof_pos": dof.copy(), "step": step})
            if step % 100 == 0:
                print(f"  step {step}/{args.num_steps}  h={root[2]:.3f}")

    return eps, label, traj, list(env.dof_names)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.variants_metadata) as f:
        meta = json.load(f)
    variant_names = list(meta.keys())
    if args.variant_name not in variant_names:
        raise ValueError(f"'{args.variant_name}' not in metadata. "
                         f"Available: {variant_names}")
    variant_idx = variant_names.index(args.variant_name)
    print(f"Variant '{args.variant_name}' (idx={variant_idx})")

    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    is_mlp = ckpt.get("model_type") == "flat_mlp"

    if is_mlp:
        eps, label, traj, dof_names = run_mlp(
            args, meta, variant_names, variant_idx, ckpt)
    else:
        eps, label, traj, dof_names = run_transformer(
            args, meta, variant_names, variant_idx, ckpt)

    # ── Stats ─────────────────────────────────────────────────────────────
    avg_ep = float(np.mean(eps))
    std_ep = float(np.std(eps))
    med_ep = float(np.median(eps))
    min_ep = int(np.min(eps))
    max_ep = int(np.max(eps))

    rng        = np.random.default_rng(42)
    boot       = [np.mean(rng.choice(eps, size=len(eps), replace=True))
                  for _ in range(10000)]
    ci_lo      = float(np.percentile(boot, 2.5))
    ci_hi      = float(np.percentile(boot, 97.5))
    k          = max(1, len(eps) // 10)
    top10      = float(np.mean(np.sort(eps)[-k:]))
    pct_500    = float(np.mean(np.array(eps) > 500) * 100)
    pct_200    = float(np.mean(np.array(eps) > 200) * 100)

    print(f"\n── Eval Results for '{args.variant_name}' ──")
    print(f"  Model     : {label}")
    print(f"  OOD       : {args.ood}")
    print(f"  Episodes  : {len(eps)}")
    print(f"  Avg       : {avg_ep:.1f} ± {std_ep:.1f} steps")
    print(f"  95% CI    : [{ci_lo:.1f}, {ci_hi:.1f}]  (±{(ci_hi-ci_lo)/2:.1f})")
    print(f"  Median    : {med_ep:.1f}  Min/Max: {min_ep}/{max_ep}")
    print(f"  Top 10%   : {top10:.1f} steps")
    print(f"  >200 steps: {pct_200:.1f}%   >500 steps: {pct_500:.1f}%")

    output = {
        "variant_name":  args.variant_name,
        "variant_idx":   variant_idx,
        "model_label":   label,
        "is_mlp":        is_mlp,
        "is_baseline":   args.baseline,
        "ood":           args.ood,
        "checkpoint":    args.checkpoint,
        "cmd":           [args.cmd_vx, args.cmd_vy, args.cmd_yaw],
        "trajectory":    traj,
        "dof_names":     dof_names,
        "eval": {
            "episode_lengths": eps,
            "avg_ep_len":      avg_ep,
            "std_ep_len":      std_ep,
            "med_ep_len":      med_ep,
            "min_ep_len":      min_ep,
            "max_ep_len":      max_ep,
            "ci_95_lo":        ci_lo,
            "ci_95_hi":        ci_hi,
            "ci_95_pm":        (ci_hi - ci_lo) / 2,
            "top10_mean":      top10,
            "pct_gt_200":      pct_200,
            "pct_gt_500":      pct_500,
            "num_rollouts":    len(eps),
            "max_ep_steps":    args.max_ep_steps,
        },
    }
    with open(args.out, "wb") as f:
        pickle.dump(output, f)
    print(f"\n  Saved → {args.out}  ({len(traj)} traj frames)")


if __name__ == "__main__":
    main()