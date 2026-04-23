"""
train_flat_mlp.py

Flat MLP RL baseline — no transformer, no morphology context, no GCN.
Trains a simple MLP actor-critic on all robots simultaneously.
Same PPO loop, same rewards, same env as ModularRunner.

This is the cleanest baseline: "why not just train a regular RL policy?"
The MLP sees concatenated proprioceptive obs from all joints and outputs
all 12 joint actions directly.

Usage:
    python train_flat_mlp.py \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --num_envs 512 --headless \
        --out_dir ./output_flat_mlp \
        --seed 1409

    # Resume
    python train_flat_mlp.py \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_variants_wide/variants_metadata.json \
        --num_envs 512 --headless \
        --out_dir ./output_flat_mlp \
        --resume ./output_flat_mlp/Apr01_12-00-00/model_200.pt
"""

# ── Isaac Gym MUST be first ───────────────────────────────────────────────────
import isaacgym
from isaacgym import gymapi
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

print("=" * 60)
print("RUNNING train_flat_mlp.py — Flat MLP, no context, no GCN")
print("=" * 60, flush=True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Reward / termination constants (same as runner.py) ────────────────────────
_REWARD_SCALES = {
    "tracking_lin_vel":   1.0,
    "tracking_ang_vel":   0.5,
    "lin_vel_z":         -2.0,
    "ang_vel_xy":        -0.05,
    "orientation":       -1.0,
    "base_height":      -10.0,
    "dof_acc":          -2.5e-7,
    "dof_vel":          -1e-3,
    "action_rate":      -0.01,
    "dof_pos_limits":   -5.0,
    "alive":             0.15,
    "hip_pos":          -1.0,
    "contact_no_vel":   -0.2,
    "feet_swing_height":-20.0,
    "contact":           0.18,
}
_SIGMA         = 0.25
_SOFT_LIMIT    = 0.9
_STANCE_THRESH = 0.55
GAIT_PERIOD    = 0.8
GAIT_OFFSET    = 0.5


def _ig_quat_to_wxyz(q):
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _quat_rot_inv(q, v):
    w = q[..., 0:1]; xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


# ── Flat obs builder ─────────────────────────────────────────────────────────

class FlatObsBuilder:
    """
    Builds a flat proprioceptive observation for the MLP policy.
    No context, no graph. Just:
        base_ang_vel (3) + proj_gravity (3) + cmd_scaled (3) +
        sin/cos phase (2) + dof_pos_norm (12) + dof_vel (12) = 35 dims
    """

    def __init__(self, env, device, variants_metadata=None):
        self.env      = env
        self.device   = device
        self.num_envs = env.num_envs

        # obs_dim: fixed for G1 12dof
        self.obs_dim    = 35   # see build()
        self.action_dim = env.num_actions   # 12

        # Feet indices for contact reward
        bnames = env.gym.get_actor_rigid_body_names(
            env.envs[0], env.actor_handles[0])
        self.left_foot_idx  = next(
            i for i, n in enumerate(bnames) if "left_ankle_roll"  in n.lower())
        self.right_foot_idx = next(
            i for i, n in enumerate(bnames) if "right_ankle_roll" in n.lower())

        # DOF limits from base robot
        dp = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
        n  = env.num_actions
        self.joint_lo = torch.tensor(
            [float(dp["lower"][i]) for i in range(n)],
            dtype=torch.float32, device=device)
        self.joint_hi = torch.tensor(
            [float(dp["upper"][i]) for i in range(n)],
            dtype=torch.float32, device=device)
        self.joint_span = (self.joint_hi - self.joint_lo).clamp(min=1e-8)

        # Per-variant base height
        if variants_metadata is not None:
            with open(variants_metadata) as f:
                meta = json.load(f)
            bht = torch.tensor(
                [m["base_height_target"] for m in meta.values()],
                dtype=torch.float32, device=device)
            self.base_height    = bht[env.env_variant_ids]
            fct = torch.tensor(
                [m.get("feet_clearance_target", 0.08) for m in meta.values()],
                dtype=torch.float32, device=device)
            self.feet_clearance = fct[env.env_variant_ids]
            self.variant_names  = list(meta.keys())
        else:
            self.base_height    = torch.full(
                (env.num_envs,), 0.78, dtype=torch.float32, device=device)
            self.feet_clearance = torch.full(
                (env.num_envs,), 0.08, dtype=torch.float32, device=device)
            self.variant_names  = ["g1"]

        print(f"[FlatObsBuilder] obs_dim={self.obs_dim}  "
              f"action_dim={self.action_dim}", flush=True)

    def build(self, commands, episode_steps, last_actions, dt):
        N   = self.num_envs
        env = self.env

        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        root_states = env.root_states[:N]
        dof_pos     = env.dof_pos[:N]
        dof_vel     = env.dof_vel[:N]

        quat_wxyz    = _ig_quat_to_wxyz(root_states[:, 3:7])
        ang_vel_w    = root_states[:, 10:13]
        base_ang_vel = _quat_rot_inv(quat_wxyz, ang_vel_w) * 0.25
        grav_w       = torch.tensor(
            [0., 0., -1.], device=self.device).expand(N, 3)
        proj_gravity = _quat_rot_inv(quat_wxyz, grav_w)
        cmd_scaled   = commands * torch.tensor(
            [2.0, 2.0, 0.25], device=self.device)

        t           = episode_steps.float() * dt
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        sin_ph      = torch.sin(2. * math.pi * phase_left).unsqueeze(1)
        cos_ph      = torch.cos(2. * math.pi * phase_left).unsqueeze(1)

        dof_pos_norm = (dof_pos - self.joint_lo) / self.joint_span * 2 - 1
        dof_vel_clip = dof_vel.clamp(-10, 10)

        obs = torch.cat([
            base_ang_vel,   # 3
            proj_gravity,   # 3
            cmd_scaled,     # 3
            sin_ph,         # 1
            cos_ph,         # 1
            dof_pos_norm,   # 12
            dof_vel_clip,   # 12
        ], dim=1)   # (N, 35)

        return obs, phase_left, phase_right


# ── MLP actor-critic ──────────────────────────────────────────────────────────

class MLPActorCritic(nn.Module):
    """
    Simple MLP actor-critic.
    Actor: obs -> 256 -> 256 -> 128 -> num_actions
    Critic: obs -> 256 -> 256 -> 128 -> 1
    """

    def __init__(self, obs_dim, action_dim, hidden=(256, 256, 128)):
        super().__init__()

        def make_mlp(out_dim):
            layers = []
            in_dim = obs_dim
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.ELU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        self.actor  = make_mlp(action_dim)
        self.critic = make_mlp(1)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Init last layers small
        nn.init.uniform_(self.actor[-1].weight,  -0.01, 0.01)
        nn.init.uniform_(self.critic[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].bias)

        total = sum(p.numel() for p in self.parameters())
        print(f"[MLPActorCritic] params={total:,}  "
              f"obs_dim={obs_dim}  action_dim={action_dim}  "
              f"hidden={hidden}", flush=True)

    def forward(self, obs):
        mu  = self.actor(obs)
        val = self.critic(obs)
        std = torch.exp(self.log_std)
        return val, Normal(mu, std)

    @torch.no_grad()
    def act(self, obs):
        val, pi = self.forward(obs)
        act  = pi.sample()
        logp = pi.log_prob(act).sum(-1, keepdim=True)
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        return self.critic(obs)


# ── Simple rollout buffer ─────────────────────────────────────────────────────

class FlatBuffer:
    def __init__(self, num_envs, obs_dim, action_dim, timesteps, device):
        self.T   = timesteps
        self.N   = num_envs
        self.dev = device

        self.obs     = torch.zeros(T, num_envs, obs_dim,    device=device)
        self.act     = torch.zeros(T, num_envs, action_dim, device=device)
        self.logp    = torch.zeros(T, num_envs, 1,          device=device)
        self.val     = torch.zeros(T, num_envs, 1,          device=device)
        self.rew     = torch.zeros(T, num_envs, 1,          device=device)
        self.mask    = torch.zeros(T, num_envs, 1,          device=device)
        self.ret     = torch.zeros(T, num_envs, 1,          device=device)
        self.adv     = torch.zeros(T, num_envs, 1,          device=device)
        self.step    = 0

    def insert(self, obs, act, logp, val, rew, mask):
        t = self.step % self.T
        self.obs[t]  = obs
        self.act[t]  = act
        self.logp[t] = logp
        self.val[t]  = val
        self.rew[t]  = rew
        self.mask[t] = mask
        self.step   += 1

    def compute_returns(self, next_val, gamma=0.99, lam=0.95):
        gae = 0.
        for t in reversed(range(self.T)):
            nv   = next_val if t == self.T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * nv * self.mask[t] - self.val[t]
            gae   = delta + gamma * lam * self.mask[t] * gae
            self.ret[t] = gae + self.val[t]
        self.adv = self.ret - self.val
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-5)

    def get_batches(self, batch_size):
        T, N = self.T, self.N
        idx  = torch.randperm(T * N, device=self.dev)
        obs  = self.obs.reshape(T * N, -1)
        act  = self.act.reshape(T * N, -1)
        logp = self.logp.reshape(T * N, 1)
        val  = self.val.reshape(T * N, 1)
        ret  = self.ret.reshape(T * N, 1)
        adv  = self.adv.reshape(T * N, 1)
        for start in range(0, T * N, batch_size):
            ids = idx[start:start + batch_size]
            yield (obs[ids], act[ids], logp[ids],
                   val[ids], ret[ids], adv[ids])


# ── Flat MLP Runner ───────────────────────────────────────────────────────────

class FlatMLPRunner:

    def __init__(self, env, xml_path, log_dir, device,
                 variants_metadata_path=None):
        from modular_policy.config import cfg
        from modular_policy.utils.meter import TrainMeter
        from modular_policy.utils import optimizer as ou

        self.cfg = cfg
        self.ou  = ou
        self.env = env
        self.device      = torch.device(device)
        self.log_dir     = log_dir
        self.writer      = None
        self.num_envs    = env.num_envs
        self.num_actions = env.num_actions
        self.dt          = env.cfg.control.decimation * env.sim_params.dt

        self.multi_variant = (variants_metadata_path is not None)

        # ── Obs builder ───────────────────────────────────────────────────
        self.obs_builder = FlatObsBuilder(
            env, device, variants_metadata=variants_metadata_path)

        self.base_height    = self.obs_builder.base_height
        self.feet_clearance = self.obs_builder.feet_clearance
        self.variant_names  = self.obs_builder.variant_names
        self.num_variants   = len(self.variant_names)

        # ── Episode state ─────────────────────────────────────────────────
        self.episode_steps   = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device)
        self.last_actions    = torch.zeros(
            self.num_envs, self.num_actions, device=self.device)
        self.last_dof_vel    = torch.zeros(
            self.num_envs, self.num_actions, device=self.device)
        self.commands        = torch.zeros(
            self.num_envs, 3, device=self.device)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # ── DOF soft limits ───────────────────────────────────────────────
        lo = self.obs_builder.joint_lo
        hi = self.obs_builder.joint_hi
        m  = (lo + hi) * 0.5
        r  = hi - lo
        self.dof_lo = (m - 0.5 * r * _SOFT_LIMIT).unsqueeze(0).expand(
            self.num_envs, -1)
        self.dof_hi = (m + 0.5 * r * _SOFT_LIMIT).unsqueeze(0).expand(
            self.num_envs, -1)

        # ── Policy ────────────────────────────────────────────────────────
        self.actor_critic = MLPActorCritic(
            obs_dim    = self.obs_builder.obs_dim,
            action_dim = self.num_actions,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS,
            weight_decay=cfg.PPO.WEIGHT_DECAY)
        self.lr_scale  = [1.0]

        # ── Buffer ────────────────────────────────────────────────────────
        global T
        T        = cfg.PPO.TIMESTEPS
        self.T   = T
        self.buf = FlatBuffer(
            self.num_envs, self.obs_builder.obs_dim,
            self.num_actions, T, self.device)

        # ── Obs normalisation (Welford) ───────────────────────────────────
        self.ob_mean  = torch.zeros(
            self.obs_builder.obs_dim, device=self.device)
        self.ob_var   = torch.ones(
            self.obs_builder.obs_dim, device=self.device)
        self.ob_count = torch.tensor(1e-4, device=self.device)
        self.clipob   = 10.0

        # ── Metrics ───────────────────────────────────────────────────────
        self.train_meter   = TrainMeter(self.variant_names)
        self.start_time    = time.time()
        self.tot_timesteps = 0
        self.resume_iter   = 0

        print("[FlatMLPRunner] Init complete.", flush=True)

    # ── Learn ─────────────────────────────────────────────────────────────────

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        from modular_policy.config import cfg
        from modular_policy.utils import optimizer as ou

        if self.log_dir and not self.writer:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            print(f"[FlatMLPRunner] TensorBoard → {self.log_dir}", flush=True)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        self.env.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs = self._get_obs_normalized()

        cfg.PPO.MAX_ITERS = num_learning_iterations
        start             = self.resume_iter

        for cur_iter in range(start, start + num_learning_iterations):
            print(f"[mlp] iter {cur_iter}", flush=True)

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break

            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr, self.lr_scale)
            t0 = time.time()

            # ── Rollout ───────────────────────────────────────────────────
            for step in range(self.T):
                val, act, logp = self.actor_critic.act(obs)
                real_actions   = act.clamp(-1., 1.)

                obs_next, rewards, dones, infos = self._step(real_actions)
                obs_next = self._normalize_obs(obs_next)

                self.train_meter.add_ep_info(infos)

                masks = torch.tensor(
                    [[0.] if d else [1.] for d in dones.cpu().tolist()],
                    dtype=torch.float32, device=self.device)

                self.buf.insert(obs, act, logp, val,
                                rewards.unsqueeze(1), masks)
                obs = obs_next

            t_rollout = time.time()

            next_val = self.actor_critic.get_value(obs)
            self.buf.compute_returns(next_val)

            self._ppo_update(cur_iter)
            t_update = time.time()

            print(f"[iter {cur_iter}] rollout={t_rollout-t0:.1f}s  "
                  f"update={t_update-t_rollout:.1f}s", flush=True)

            self.train_meter.update_mean()
            self.tot_timesteps += self.T * self.num_envs

            if cur_iter % cfg.LOG_PERIOD == 0 and cfg.LOG_PERIOD > 0:
                self._log(cur_iter, start + num_learning_iterations)

            if cur_iter % cfg.CHECKPOINT_PERIOD == 0:
                self.save(cur_iter)

        self.save(start + num_learning_iterations)
        print("[FlatMLPRunner] Training complete.")

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(self, cur_iter):
        from modular_policy.config import cfg

        for epoch in range(cfg.PPO.EPOCHS):
            for (obs, act, logp_old, val_old,
                 ret, adv) in self.buf.get_batches(cfg.PPO.BATCH_SIZE):

                val_new, pi = self.actor_critic.forward(obs)
                logp_new    = pi.log_prob(act).sum(-1, keepdim=True)
                entropy     = pi.entropy().mean()

                ratio     = torch.exp(logp_new - logp_old)
                approx_kl = (logp_old - logp_new).mean().item()

                if (cfg.PPO.KL_TARGET_COEF is not None and
                        approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01):
                    print(f"  Early stop iter {cur_iter} epoch {epoch} "
                          f"kl={approx_kl:.4f}")
                    return

                surr1   = ratio * adv
                surr2   = torch.clamp(
                    ratio, 1. - cfg.PPO.CLIP_EPS,
                           1. + cfg.PPO.CLIP_EPS) * adv
                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.PPO.USE_CLIP_VALUE_FUNC:
                    vclip = val_old + (val_new - val_old).clamp(
                        -cfg.PPO.CLIP_EPS, cfg.PPO.CLIP_EPS)
                    vl    = 0.5 * torch.max(
                        (val_new - ret).pow(2),
                        (vclip   - ret).pow(2)).mean()
                else:
                    vl = 0.5 * (ret - val_new).pow(2).mean()

                loss = (vl * cfg.PPO.VALUE_COEF
                        + pi_loss
                        - entropy * cfg.PPO.ENTROPY_COEF)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM)
                self.optimizer.step()

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, cur_iter, path=None):
        if path is None and self.log_dir:
            path = os.path.join(self.log_dir, f"model_{cur_iter}.pt")
        if path:
            torch.save({
                "model_state_dict":     self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter":                 cur_iter,
                "ob_mean":              self.ob_mean.cpu(),
                "ob_var":               self.ob_var.cpu(),
                "ob_count":             self.ob_count.cpu(),
                "model_type":           "flat_mlp",
                "obs_dim":              self.obs_builder.obs_dim,
                "action_dim":           self.num_actions,
            }, path)
            print(f"[FlatMLPRunner] Saved → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["model_state_dict"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[FlatMLPRunner] Optimizer load skipped: {e}")
        if "ob_mean" in ckpt:
            self.ob_mean  = ckpt["ob_mean"].to(self.device)
            self.ob_var   = ckpt["ob_var"].to(self.device)
            self.ob_count = ckpt["ob_count"].to(self.device)
        self.resume_iter = ckpt.get("iter", 0)
        print(f"[FlatMLPRunner] Loaded: {path}  iter={self.resume_iter}")

    # ── Environment ───────────────────────────────────────────────────────────

    def _step(self, real_actions):
        from isaacgym import gymtorch

        self.last_dof_vel = self.env.dof_vel.clone()
        self.env.actions  = real_actions

        for _ in range(self.env.cfg.control.decimation):
            self.env.torques = self.env._compute_torques(self.env.actions)
            self.env.gym.set_dof_actuation_force_tensor(
                self.env.sim, gymtorch.unwrap_tensor(self.env.torques))
            self.env.gym.simulate(self.env.sim)
            self.env.gym.fetch_results(self.env.sim, True)
            self.env.gym.refresh_dof_state_tensor(self.env.sim)

        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_net_contact_force_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)

        obs, phase_left, phase_right = self.obs_builder.build(
            self.commands, self.episode_steps, self.last_actions, self.dt)

        rewards = self._compute_rewards(real_actions, phase_left, phase_right)
        dones   = self._check_termination()

        self.episode_steps   += 1
        self.episode_returns += rewards
        self.last_actions     = real_actions.clone()

        infos    = self._build_infos(dones)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids):
            self.env.reset_idx(done_ids)
            self._resample_commands(done_ids)
            self.episode_steps[done_ids]   = 0
            self.episode_returns[done_ids] = 0
            self.last_actions[done_ids]    = 0
            self.last_dof_vel[done_ids]    = 0

        return obs, rewards, dones, infos

    def _get_obs_normalized(self):
        self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
        self.env.gym.refresh_dof_state_tensor(self.env.sim)
        self.env.gym.refresh_rigid_body_state_tensor(self.env.sim)
        obs, _, _ = self.obs_builder.build(
            self.commands, self.episode_steps, self.last_actions, self.dt)
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs):
        bm  = obs.mean(0); bv = obs.var(0); bc = torch.tensor(
            obs.shape[0], dtype=torch.float32, device=self.device)
        d   = bm - self.ob_mean
        tot = self.ob_count + bc
        self.ob_mean  = self.ob_mean + d * bc / tot
        self.ob_var   = (self.ob_var * self.ob_count + bv * bc +
                         d.pow(2) * self.ob_count * bc / tot) / tot
        self.ob_count = tot
        return ((obs - self.ob_mean) /
                (self.ob_var + 1e-8).sqrt()).clamp(-self.clipob, self.clipob)

    def _compute_rewards(self, actions, phase_left, phase_right):
        dt = self.dt; s = _REWARD_SCALES; N = self.num_envs

        root      = self.env.root_states[:N]
        q_wxyz    = _ig_quat_to_wxyz(root[:, 3:7])
        height    = root[:, 2]
        base_lin  = _quat_rot_inv(q_wxyz, root[:, 7:10])
        base_ang  = _quat_rot_inv(q_wxyz, root[:, 10:13])
        grav_w    = torch.tensor([0.,0.,-1.], device=self.device).expand(N,3)
        proj_grav = _quat_rot_inv(q_wxyz, grav_w)
        dof_pos   = self.env.dof_pos[:N]
        dof_vel   = self.env.dof_vel[:N]
        rb        = self.env.rigid_body_states_view[:N]

        fi      = (self.obs_builder.left_foot_idx,
                   self.obs_builder.right_foot_idx)
        lc      = (self.env.contact_forces[:N, fi[0], 2].abs() > 1.).float()
        rc      = (self.env.contact_forces[:N, fi[1], 2].abs() > 1.).float()
        lz      = rb[:, fi[0], 2]; rz = rb[:, fi[1], 2]
        lv      = rb[:, fi[0], 7:10]; rv = rb[:, fi[1], 7:10]

        r_tlv  = torch.exp(-((self.commands[:,:2]-base_lin[:,:2])**2).sum(1)/_SIGMA)
        r_tav  = torch.exp(-((self.commands[:,2] -base_ang[:,2])**2)/_SIGMA)
        r_lvz  = base_lin[:,2]**2
        r_avxy = (base_ang[:,:2]**2).sum(1)
        r_ori  = (proj_grav[:,:2]**2).sum(1)
        r_bh   = (height - self.base_height)**2
        r_dacc = ((dof_vel - self.last_dof_vel)/dt).pow(2).sum(1)
        r_dv   = dof_vel.pow(2).sum(1)
        r_ar   = (actions - self.last_actions).pow(2).sum(1)
        out_lo = (self.dof_lo - dof_pos).clamp(min=0.)
        out_hi = (dof_pos - self.dof_hi).clamp(min=0.)
        r_dpl  = (out_lo + out_hi).sum(1)
        r_alive= torch.ones(N, device=self.device)
        r_hip  = dof_pos[:,[0,1,6,7]].pow(2).sum(1)
        ls     = (phase_left  < _STANCE_THRESH).float()
        rs     = (phase_right < _STANCE_THRESH).float()
        r_con  = (lc==ls).float() + (rc==rs).float()
        r_cnv  = lv.pow(2).sum(1)*lc + rv.pow(2).sum(1)*rc
        fc     = self.feet_clearance
        r_fsh  = (lz-fc).pow(2)*(1-lc) + (rz-fc).pow(2)*(1-rc)

        return (
            s["tracking_lin_vel"]  * r_tlv  * dt +
            s["tracking_ang_vel"]  * r_tav  * dt +
            s["lin_vel_z"]         * r_lvz  * dt +
            s["ang_vel_xy"]        * r_avxy * dt +
            s["orientation"]       * r_ori  * dt +
            s["base_height"]       * r_bh   * dt +
            s["dof_acc"]           * r_dacc * dt +
            s["dof_vel"]           * r_dv   * dt +
            s["action_rate"]       * r_ar   * dt +
            s["dof_pos_limits"]    * r_dpl  * dt +
            s["alive"]             * r_alive* dt +
            s["hip_pos"]           * r_hip  * dt +
            s["contact"]           * r_con  * dt +
            s["contact_no_vel"]    * r_cnv  * dt +
            s["feet_swing_height"] * r_fsh  * dt
        )

    def _check_termination(self):
        root  = self.env.root_states[:self.num_envs]
        h     = root[:, 2]
        q     = _ig_quat_to_wxyz(root[:, 3:7])
        w,x,y,z = q[:,0],q[:,1],q[:,2],q[:,3]
        roll  = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
        pf    = self.env.contact_forces[:self.num_envs, 0, 2].abs()
        h_lo  = self.base_height * 0.5
        h_hi  = self.base_height * 1.5
        return ((pf>1.) | (h<h_lo) | (h>h_hi) |
                (pitch.abs()>1.) | (roll.abs()>.8) |
                (self.episode_steps >= 1000))

    def _resample_commands(self, env_ids):
        n = len(env_ids)
        self.commands[env_ids,0] = torch.FloatTensor(n).uniform_(0., 1.).to(self.device)
        self.commands[env_ids,1] = torch.FloatTensor(n).uniform_(-.5,.5).to(self.device)
        self.commands[env_ids,2] = torch.FloatTensor(n).uniform_(-.5,.5).to(self.device)
        small = self.commands[env_ids,:2].norm(dim=1) < .2
        self.commands[env_ids[small],:2] = 0.

    def _build_infos(self, dones):
        infos   = []
        done_np = dones.cpu().numpy()
        for i in range(self.num_envs):
            info = {"name": "g1"}
            if self.multi_variant:
                vid  = int(self.env.env_variant_ids[i].item())
                info["name"] = self.variant_names[vid]
            if done_np[i]:
                info["episode"] = {
                    "r": float(self.episode_returns[i].item()),
                    "l": int(self.episode_steps[i].item()),
                }
            infos.append(info)
        return infos

    def _log(self, cur_iter, total_iters):
        elapsed = time.time() - self.start_time
        fps     = int(self.tot_timesteps / max(elapsed, 1))
        eta     = elapsed / max(cur_iter+1,1) * (total_iters - cur_iter)
        print(f"\nIter {cur_iter}/{total_iters} | "
              f"timesteps {self.tot_timesteps} | FPS {fps} | ETA {eta:.0f}s")
        self.train_meter.log_stats()
        if not self.writer:
            return
        self.writer.add_scalar("Perf/fps",        fps,                 cur_iter)
        self.writer.add_scalar("Perf/timesteps",   self.tot_timesteps,  cur_iter)
        self.writer.add_scalar("Train/lr",
            self.optimizer.param_groups[0]["lr"], cur_iter)

        all_returns = []
        all_ep_lens = []
        for name, meter in self.train_meter.agent_meters.items():
            if len(meter.ep_rew["reward"]) > 0:
                mr = float(np.mean(meter.ep_rew["reward"]))
                self.writer.add_scalar(f"Reward/variant_{name}", mr, cur_iter)
                all_returns.extend(list(meter.ep_rew["reward"]))
            if len(meter.ep_len) > 0:
                ml = float(np.mean(meter.ep_len))
                ema = float(meter.ep_len_ema) if meter.ep_len_ema > 0 else ml
                self.writer.add_scalar(f"EpLen/variant_{name}",     ml,  cur_iter)
                self.writer.add_scalar(f"EpLen_EMA/variant_{name}", ema, cur_iter)
                all_ep_lens.append(ml)
            self.writer.add_scalar(
                f"EpCount/variant_{name}", meter.ep_count, cur_iter)

        if all_returns:
            self.writer.add_scalar("Reward/mean_all_variants",
                                   float(np.mean(all_returns)), cur_iter)
        if all_ep_lens:
            self.writer.add_scalar("EpLen/mean_all_variants",
                                   float(np.mean(all_ep_lens)), cur_iter)
        self.writer.flush()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_path",          required=True)
    p.add_argument("--variants_metadata", default=None)
    p.add_argument("--num_envs",          type=int,   default=512)
    p.add_argument("--out_dir",           default="./output_flat_mlp")
    p.add_argument("--sim_device",        default="cuda:0")
    p.add_argument("--rl_device",         default="cuda:0")
    p.add_argument("--headless",          action="store_true", default=True)
    p.add_argument("--resume",            default=None)
    p.add_argument("--seed",              type=int, default=1409)
    p.add_argument("--max_iters",         type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()

    from modular_policy.config import cfg
    from legged_gym.envs.g1.g1_config import G1RoughCfg
    from legged_gym.utils.helpers import parse_sim_params, class_to_dict, set_seed

    set_seed(args.seed)
    cfg.PPO.NUM_ENVS             = args.num_envs
    cfg.PPO.MAX_ITERS            = args.max_iters
    cfg.PPO.EARLY_EXIT_MAX_ITERS = args.max_iters
    cfg.DEVICE                   = args.rl_device
    cfg.OUT_DIR                  = args.out_dir

    # Needed for TrainMeter
    if args.variants_metadata:
        with open(args.variants_metadata) as f:
            meta = json.load(f)
        cfg.ENV.WALKERS = list(meta.keys())
    else:
        cfg.ENV.WALKERS = ["g1"]

    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    env_cfg              = G1RoughCfg()
    env_cfg.env.num_envs = args.num_envs

    class _Args:
        physics_engine    = gymapi.SIM_PHYSX
        sim_device        = args.sim_device
        rl_device         = args.rl_device
        headless          = args.headless
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

    if args.variants_metadata:
        from legged_gym.envs.g1.multi_variant_env import MultiVariantG1Robot
        env = MultiVariantG1Robot(
            cfg                    = env_cfg,
            sim_params             = sim_params,
            physics_engine         = gymapi.SIM_PHYSX,
            sim_device             = args.sim_device,
            headless               = args.headless,
            variants_metadata_path = args.variants_metadata,
        )
    else:
        from legged_gym.envs.g1.g1_env import G1Robot
        env = G1Robot(
            cfg            = env_cfg,
            sim_params     = sim_params,
            physics_engine = gymapi.SIM_PHYSX,
            sim_device     = args.sim_device,
            headless       = args.headless,
        )

    print(f"Env ready — {env.num_envs} envs  {env.num_actions} actions")

    log_dir = os.path.join(cfg.OUT_DIR, datetime.now().strftime("%b%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = FlatMLPRunner(
        env                    = env,
        xml_path               = os.path.abspath(args.xml_path),
        log_dir                = log_dir,
        device                 = args.rl_device,
        variants_metadata_path = args.variants_metadata,
    )

    if args.resume:
        runner.load(args.resume)

    runner.learn(
        num_learning_iterations = cfg.PPO.MAX_ITERS,
        init_at_random_ep_len   = True)


if __name__ == "__main__":
    main()