"""
modular_policy/algos/ppo/runner.py

Full PPO training loop.
Uses G1Robot for physics, ObsBuilder for obs, ActorCritic + Buffer for policy.
Drop-in replacement for OnPolicyRunner — same .learn() signature.
"""

import os
import time
import math
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic, Agent
from modular_policy.algos.ppo.buffer import Buffer
from modular_policy.algos.ppo.obs_builder import ObsBuilder
from modular_policy.utils.meter import TrainMeter
from modular_policy.utils import optimizer as ou


# ── Reward constants ──────────────────────────────────────────────────────────
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
_BASE_HEIGHT  = 0.78
_SIGMA        = 0.25
_SOFT_LIMIT   = 0.9
_STANCE_THRESH = 0.55


def _ig_quat_to_wxyz(q):
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _quat_rot_inv(q, v):
    w   = q[..., 0:1]; xyz = q[..., 1:]
    t   = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


class ModularRunner:
    """Replaces OnPolicyRunner. Call .learn() to train."""

    def __init__(self, env, xml_path, log_dir=None, device="cuda:0"):
        print("[ModularRunner] Starting __init__...", flush=True)
        self.env     = env
        self.device  = torch.device(device)
        self.log_dir = log_dir
        self.writer  = None

        self.num_envs    = env.num_envs
        self.num_actions = env.num_actions   # 12
        self.dt          = env.cfg.control.decimation * env.sim_params.dt

        # Update cfg so model sizes are consistent
        cfg.PPO.NUM_ENVS = self.num_envs

        # ── Obs builder ───────────────────────────────────────────────────
        print("[ModularRunner] Creating ObsBuilder...", flush=True)
        self.obs_builder = ObsBuilder(env, xml_path, device)
        print("[ModularRunner] ObsBuilder Done", flush=True)

        # ── Episode state ─────────────────────────────────────────────────
        print("[ModularRunner] Initialising episode state tensors...", flush=True)
        self.episode_steps   = torch.zeros(self.num_envs, dtype=torch.long,    device=self.device)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_actions    = torch.zeros(self.num_envs, self.num_actions,    device=self.device)
        self.last_dof_vel    = torch.zeros(self.num_envs, self.num_actions,    device=self.device)
        self.commands        = torch.zeros(self.num_envs, 3,                   device=self.device)
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

        # Soft DOF limits
        print("[ModularRunner] Building DOF limits...", flush=True)
        self._build_dof_limits()

        # ── Policy ────────────────────────────────────────────────────────
        print("[ModularRunner] Creating ActorCritic...", flush=True)
        self.actor_critic = ActorCritic(
            self.obs_builder.observation_space,
            self.obs_builder.action_space,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.actor_critic.parameters())
        print(f"[ModularRunner] Policy params: {total_params:,}")

        self.agent = Agent(self.actor_critic)

        # ── Buffer ────────────────────────────────────────────────────────
        print("[ModularRunner] Creating Buffer...", flush=True)
        self.buffer = Buffer(
            self.obs_builder.observation_space,
            self.obs_builder.action_space.shape,
        )
        print("[ModularRunner] Buffer to device...", flush=True)
        self.buffer.to(self.device)

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS,
            weight_decay=cfg.PPO.WEIGHT_DECAY)
        self.lr_scale = [1.0]

        # ── Metrics ───────────────────────────────────────────────────────
        self.train_meter  = TrainMeter(["g1"])
        self.start_time   = time.time()
        self.tot_timesteps = 0

        # Online obs normalisation (Welford)
        prop_dim = cfg.MODEL.MAX_LIMBS * self.obs_builder.limb_obs_size
        self.ob_mean  = torch.zeros(prop_dim, device=self.device)
        self.ob_var   = torch.ones( prop_dim, device=self.device)
        self.ob_count = 1e-4
        self.clipob   = 10.0

        print("[ModularRunner] Init complete.", flush=True)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        print("[learn] Starting...", flush=True)
        if self.log_dir and not self.writer:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        # Initial reset and obs
        print("[learn] Resetting envs...", flush=True)
        self.env.reset_idx(torch.arange(self.num_envs, device=self.device))
        print("[learn] Getting initial obs...", flush=True)
        obs = self._get_obs_normalized()

        print("[learn] Got obs, starting training loop...", flush=True)
        cfg.PPO.MAX_ITERS = num_learning_iterations   # needed for LR schedule

        for cur_iter in range(num_learning_iterations):
            print(f"[learn] iter {cur_iter}", flush=True)

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break

            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr, self.lr_scale)

            # ── Rollout ───────────────────────────────────────────────────
            import time
            for step in range(cfg.PPO.TIMESTEPS):
                unimal_ids = [0] * self.num_envs
                t0 = time.time()

                val, act, logp, dmv, dmmu = self.agent.act(
                    obs, unimal_ids=unimal_ids)
                t1 = time.time()

                act_mask     = self.obs_builder.act_padding_mask[0].bool()
                real_actions = act[:, ~act_mask].clamp(-1., 1.)

                obs_next, rewards, dones, infos = self._step(real_actions)
                t2 = time.time()

                obs_next = self._normalize_obs(obs_next)
                t3 = time.time()

                if step < 3:
                    print(f"  [step {step}] act={t1-t0:.3f}s  env_step={t2-t1:.3f}s  norm={t3-t2:.3f}s", flush=True)

                # Strip action padding
                act_mask     = self.obs_builder.act_padding_mask[0].bool()
                real_actions = act[:, ~act_mask].clamp(-1., 1.)   # (N, 12)

                obs_next, rewards, dones, infos = self._step(real_actions)
                obs_next = self._normalize_obs(obs_next)

                self.train_meter.add_ep_info(infos)

                masks = torch.tensor(
                    [[0.] if d else [1.] for d in dones.cpu().tolist()],
                    dtype=torch.float32, device=self.device)
                timeouts = torch.ones_like(masks)

                self.buffer.insert(
                    obs, act, logp, val,
                    rewards.unsqueeze(1), masks, timeouts,
                    dmv, dmmu, unimal_ids)
                obs = obs_next

            t_rollout = time.time()
            # ── Value bootstrap ───────────────────────────────────────────
            next_val = self.agent.get_value(obs, unimal_ids=[0]*self.num_envs)
            self.buffer.compute_returns(next_val)
            t_bootstrap = time.time()

            # ── PPO update ────────────────────────────────────────────────
            self._ppo_update(cur_iter)
            t_update = time.time()

            print(f"[iter {cur_iter}] rollout={t_rollout-t0:.1f}s  bootstrap={t_bootstrap-t_rollout:.1f}s  update={t_update-t_bootstrap:.1f}s", flush=True)

            self.train_meter.update_mean()
            self.tot_timesteps += cfg.PPO.TIMESTEPS * self.num_envs

            if cur_iter % cfg.LOG_PERIOD == 0 and cfg.LOG_PERIOD > 0:
                self._log(cur_iter, num_learning_iterations)

            if cur_iter % cfg.CHECKPOINT_PERIOD == 0:
                self.save(cur_iter)

        self.save(num_learning_iterations)
        print("Training complete.")

    def save(self, cur_iter, path=None):
        if path is None and self.log_dir:
            path = os.path.join(self.log_dir, f"model_{cur_iter}.pt")
        if path:
            torch.save({
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": cur_iter,
                "ob_mean": self.ob_mean.cpu(),
                "ob_var":  self.ob_var.cpu(),
                "ob_count": self.ob_count,
            }, path)
            print(f"[ModularRunner] Saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ob_mean" in ckpt:
            self.ob_mean  = ckpt["ob_mean"].to(self.device)
            self.ob_var   = ckpt["ob_var"].to(self.device)
            self.ob_count = ckpt["ob_count"]
        print(f"[ModularRunner] Loaded: {path}")

    # ─────────────────────────────────────────────────────────────────────
    # Environment stepping
    # ─────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────
    # Rewards
    # ─────────────────────────────────────────────────────────────────────

    def _compute_rewards(self, actions, phase_left, phase_right):
        dt = self.dt
        s  = _REWARD_SCALES
        N  = self.num_envs

        root  = self.env.root_states[:N]
        q_wxyz = _ig_quat_to_wxyz(root[:, 3:7])
        height = root[:, 2]

        base_lin = _quat_rot_inv(q_wxyz, root[:, 7:10])
        base_ang = _quat_rot_inv(q_wxyz, root[:, 10:13])

        grav_w   = torch.tensor([0., 0., -1.], device=self.device).expand(N, 3)
        proj_grav = _quat_rot_inv(q_wxyz, grav_w)

        dof_pos  = self.env.dof_pos[:N]
        dof_vel  = self.env.dof_vel[:N]
        rb       = self.env.rigid_body_states_view[:N]

        fi = self.obs_builder.feet_indices
        left_fz  = self.env.contact_forces[:N, fi[0], 2].abs()
        right_fz = self.env.contact_forces[:N, fi[1], 2].abs()
        lc = (left_fz  > 1.).float()
        rc = (right_fz > 1.).float()

        lz = rb[:, fi[0], 2];  rz = rb[:, fi[1], 2]
        lv = rb[:, fi[0], 7:10]; rv = rb[:, fi[1], 7:10]

        r_tlv  = torch.exp(-((self.commands[:,:2] - base_lin[:,:2])**2).sum(1) / _SIGMA)
        r_tav  = torch.exp(-((self.commands[:,2]  - base_ang[:,2])**2) / _SIGMA)
        r_lvz  = base_lin[:,2]**2
        r_avxy = (base_ang[:,:2]**2).sum(1)
        r_ori  = (proj_grav[:,:2]**2).sum(1)
        r_bh   = (height - _BASE_HEIGHT)**2
        r_dacc = ((dof_vel - self.last_dof_vel) / dt).pow(2).sum(1)
        r_dv   = dof_vel.pow(2).sum(1)
        r_ar   = (actions - self.last_actions).pow(2).sum(1)

        out_lo = (self.dof_lo - dof_pos).clamp(min=0.)
        out_hi = (dof_pos - self.dof_hi).clamp(min=0.)
        r_dpl  = (out_lo + out_hi).sum(1)

        r_alive = torch.ones(N, device=self.device)
        r_hip   = dof_pos[:, [0,1,6,7]].pow(2).sum(1)

        ls = (phase_left  < _STANCE_THRESH).float()
        rs = (phase_right < _STANCE_THRESH).float()
        r_con = (lc == ls).float() + (rc == rs).float()

        r_cnv  = (lv.pow(2).sum(1)*lc + rv.pow(2).sum(1)*rc)
        r_fsh  = ((lz-0.08).pow(2)*(1-lc) + (rz-0.08).pow(2)*(1-rc))

        total = (
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
        return total

    # ─────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────

    def _check_termination(self):
        root   = self.env.root_states[:self.num_envs]
        height = root[:, 2]
        q      = _ig_quat_to_wxyz(root[:, 3:7])
        w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
        roll  = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = torch.asin((2*(w*y-z*x)).clamp(-1,1))
        pelvis_fz = self.env.contact_forces[:self.num_envs, 0, 2].abs()
        return (pelvis_fz > 1.) | (height < .4) | (height > 1.15) | \
               (pitch.abs() > 1.) | (roll.abs() > .8) | (self.episode_steps >= 1000)

    # ─────────────────────────────────────────────────────────────────────
    # PPO update
    # ─────────────────────────────────────────────────────────────────────

    def _ppo_update(self, cur_iter):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for epoch_i in range(cfg.PPO.EPOCHS):
            for batch in self.buffer.get_sampler(adv):
                unimal_ids = [0] * len(batch["act"])

                val, _, logp, ent, _, _ = self.actor_critic(
                    batch["obs"], batch["act"],
                    dropout_mask_v=batch["dropout_mask_v"],
                    dropout_mask_mu=batch["dropout_mask_mu"],
                    unimal_ids=batch["unimal_ids"])

                ratio     = torch.exp(logp - batch["logp_old"])
                approx_kl = (batch["logp_old"] - logp).mean().item()

                if (cfg.PPO.KL_TARGET_COEF is not None and
                        approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01):
                    print(f"  Early stop iter {cur_iter} epoch {epoch_i+1} "
                          f"kl={approx_kl:.4f}")
                    return

                surr1 = ratio * batch["adv"]
                surr2 = torch.clamp(
                    ratio, 1. - cfg.PPO.CLIP_EPS,
                           1. + cfg.PPO.CLIP_EPS) * batch["adv"]
                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.PPO.USE_CLIP_VALUE_FUNC:
                    vclip   = batch["val"] + (val - batch["val"]).clamp(
                        -cfg.PPO.CLIP_EPS, cfg.PPO.CLIP_EPS)
                    vl      = 0.5 * torch.max(
                        (val - batch["ret"]).pow(2),
                        (vclip - batch["ret"]).pow(2)).mean()
                else:
                    vl = 0.5 * (batch["ret"] - val).pow(2).mean()

                loss = vl * cfg.PPO.VALUE_COEF + pi_loss - ent * cfg.PPO.ENTROPY_COEF
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM)
                self.optimizer.step()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_dof_limits(self):
        dp = self.env.gym.get_actor_dof_properties(
            self.env.envs[0], self.env.actor_handles[0])
        lo = torch.tensor([float(dp["lower"][i]) for i in range(self.num_actions)],
                          dtype=torch.float32, device=self.device)
        hi = torch.tensor([float(dp["upper"][i]) for i in range(self.num_actions)],
                          dtype=torch.float32, device=self.device)
        m  = (lo + hi) * .5;  r = hi - lo
        self.dof_lo = m - .5 * r * _SOFT_LIMIT
        self.dof_hi = m + .5 * r * _SOFT_LIMIT

    def _resample_commands(self, env_ids):
        n = len(env_ids)
        self.commands[env_ids, 0] = torch.FloatTensor(n).uniform_(0., 1.).to(self.device)
        self.commands[env_ids, 1] = torch.FloatTensor(n).uniform_(-.5, .5).to(self.device)
        self.commands[env_ids, 2] = torch.FloatTensor(n).uniform_(-.5, .5).to(self.device)
        small = self.commands[env_ids, :2].norm(dim=1) < .2
        self.commands[env_ids[small], :2] = 0.

    def _normalize_obs(self, obs):
        prop = obs["proprioceptive"]
        bm   = prop.mean(0); bv = prop.var(0); bc = prop.shape[0]
        d    = bm - self.ob_mean
        tot  = self.ob_count + bc
        self.ob_mean  = self.ob_mean + d * bc / tot
        self.ob_var   = (self.ob_var * self.ob_count + bv * bc +
                         d.pow(2) * self.ob_count * bc / tot) / tot
        self.ob_count = tot
        obs["proprioceptive"] = (
            (prop - self.ob_mean) / (self.ob_var + 1e-8).sqrt()
        ).clamp(-self.clipob, self.clipob)
        return obs

    def _build_infos(self, dones):
        infos   = []
        done_np = dones.cpu().numpy()
        for i in range(self.num_envs):
            info = {"name": "g1"}
            if done_np[i]:
                info["episode"] = {
                    "r": float(self.episode_returns[i].item()),
                    "l": int(self.episode_steps[i].item()),
                }
            infos.append(info)
        return infos

    def _log(self, cur_iter, total_iters):
        elapsed = time.time() - self.start_time
        fps     = int(self.tot_timesteps / elapsed)
        eta     = elapsed / (cur_iter + 1) * (total_iters - cur_iter)
        print(f"\nIter {cur_iter}/{total_iters} | "
              f"timesteps {self.tot_timesteps} | FPS {fps} | ETA {eta:.0f}s")
        self.train_meter.log_stats()
        if self.writer:
            self.writer.add_scalar("Perf/fps",         fps, cur_iter)
            self.writer.add_scalar("Perf/timesteps",   self.tot_timesteps, cur_iter)
            self.writer.add_scalar("Train/lr",
                self.optimizer.param_groups[0]["lr"], cur_iter)