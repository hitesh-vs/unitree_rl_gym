"""
deploy/deploy_mujoco/deploy_mujoco_film_rwse.py

Tests FiLM + RWSE + GCN policy on G1 variants (seen or OOD) in MuJoCo.
Loads from inference package (model_400_inference.pt) which contains
all normalization stats and config baked in.

Two-path loading (same as training):
  --xml_path  : stripped MuJoCo XML — used for simulation physics + graph topology
  --urdf_path : variant URDF — used ONLY for 12-dim context vector

Usage (seen variant):
    python deploy/deploy_mujoco/deploy_mujoco_film_rwse.py \
        --inference_pkg output_film_wide/Mar31_18-18-17/model_400_inference.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --urdf_path resources/robots/g1_variants_wide/robot_variant_3.urdf \
        --variant_name robot_variant_3 \
        --output_file trajectory_variant3.pkl \
        --duration 10.0 --cmd_vx 0.5

Usage (OOD heldout variant):
    python deploy/deploy_mujoco/deploy_mujoco_film_rwse.py \
        --inference_pkg output_film_wide/Mar31_18-18-17/model_400_inference.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --urdf_path resources/robots/g1_variants_heldout/robot_variant_0.urdf \
        --variant_name robot_variant_0_heldout \
        --output_file trajectory_heldout0.pkl \
        --duration 10.0 --cmd_vx 0.5
"""

import os
import sys
import math
import time
import pickle
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic, Agent
from modular_policy.algos.ppo.obs_builder import (
    _load_variant_graph, _BoxSpace, _DictSpace)

# ── Constants (must match training) ──────────────────────────────────────────
GAIT_PERIOD  = 0.8
GAIT_OFFSET  = 0.5
SIM_DT       = 0.005
DECIMATION   = 4
POLICY_DT    = SIM_DT * DECIMATION   # 0.02s

KP = {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40}
KD = {"hip_yaw": 2,   "hip_roll": 2,   "hip_pitch": 2,   "knee": 4,   "ankle": 2}
ACTION_SCALE = 0.25

DEFAULT_ANGLES = {
    "left_hip_yaw_joint":      0.0,
    "left_hip_roll_joint":     0.0,
    "left_hip_pitch_joint":   -0.1,
    "left_knee_joint":         0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint":   0.0,
    "right_hip_yaw_joint":     0.0,
    "right_hip_roll_joint":    0.0,
    "right_hip_pitch_joint":  -0.1,
    "right_knee_joint":        0.3,
    "right_ankle_pitch_joint":-0.2,
    "right_ankle_roll_joint":  0.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def quat_rotate_inverse_np(q, v):
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v - w * t + np.cross(np.array([x, y, z]), t)


def quat_to_expmap(q):
    w        = np.clip(q[0], -1 + 1e-7, 1 - 1e-7)
    xyz      = q[1:]
    angle    = 2.0 * np.arccos(abs(w))
    sin_half = np.sin(angle / 2)
    if sin_half < 1e-8:
        return np.zeros(3)
    return xyz / sin_half * angle


def get_kp_kd_arrays(m):
    kp_arr = np.zeros(12, dtype=np.float32)
    kd_arr = np.zeros(12, dtype=np.float32)
    for i in range(12):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        if jname is None:
            continue
        jname = jname.lower()
        for key in KP:
            if key in jname:
                kp_arr[i] = KP[key]
                kd_arr[i] = KD[key]
                break
    return kp_arr, kd_arr


def get_default_angles(m):
    default = np.zeros(12, dtype=np.float32)
    for i in range(12):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        if jname and jname in DEFAULT_ANGLES:
            default[i] = DEFAULT_ANGLES[jname]
    return default


def reset_sim(m, d, default_angles, base_height=0.8):
    mujoco.mj_resetData(m, d)
    d.qpos[2]  = base_height
    d.qpos[3]  = 1.0
    d.qpos[7:] = default_angles
    d.qvel[:]  = 0.
    mujoco.mj_forward(m, d)


# ── Obs builder ───────────────────────────────────────────────────────────────

class DeployObsBuilder:
    def __init__(self, xml_path, device, urdf_path=None):
        self.device        = device
        self.max_limbs     = cfg.MODEL.MAX_LIMBS
        self.max_joints    = cfg.MODEL.MAX_JOINTS
        self.graph_enc     = cfg.MODEL.GRAPH_ENCODING
        self.use_ltv       = False
        self.limb_obs_size = 26

        print(f"[DeployObsBuilder] Loading variant graph ...")
        print(f"  xml  (topology+sim): {os.path.basename(xml_path)}")
        print(f"  urdf (context only): "
              f"{os.path.basename(urdf_path) if urdf_path else 'none (using xml)'}")

        vd = _load_variant_graph(
            xml_path, self.max_limbs, self.max_joints,
            self.graph_enc, self.use_ltv, 12, device,
            urdf_path=urdf_path)

        self.num_limbs        = vd.num_limbs
        self.context          = vd.context
        self.edges            = vd.edges
        self.traversals       = vd.traversals
        self.swat_re          = vd.swat_re
        self.obs_padding_mask = vd.obs_padding_mask
        self.act_padding_mask = vd.act_padding_mask
        self.joint_lo         = vd.joint_lo.cpu().numpy()
        self.joint_span       = vd.joint_span.cpu().numpy()
        self.jrange_norm      = vd.jrange_norm.cpu().numpy()

        if self.graph_enc != "none":
            self.graph_node_features = vd.graph_node_features
            self.graph_A_norm        = vd.graph_A_norm
        else:
            self.graph_node_features = None
            self.graph_A_norm        = None

        print(f"[DeployObsBuilder] Ready: {self.num_limbs} limbs, "
              f"context_shape={self.context.shape}")

    def build(self, d, command, episode_step, last_action):
        # Root state — XML has floating base
        quat_wxyz    = d.qpos[3:7]
        ang_vel_w    = d.qvel[3:6]
        base_ang_vel = ang_vel_w * 0.25   # already in body frame in MuJoCo
        qw, qx, qy, qz = d.qpos[3:7]
        proj_grav = np.array([
            2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz)
        ])
        cmd_scaled   = command * np.array([2.0, 2.0, 0.25])

        t            = episode_step * POLICY_DT
        phase_left   = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right  = (phase_left + GAIT_OFFSET) % 1.0
        sin_ph       = np.sin(2. * math.pi * phase_left)
        cos_ph       = np.cos(2. * math.pi * phase_left)
        global_state = np.concatenate(
            [base_ang_vel, proj_grav, cmd_scaled, [sin_ph], [cos_ph]])  # (11,)

        dof_pos    = d.qpos[7:].copy()
        angle_norm = (dof_pos - self.joint_lo) / self.joint_span

        pelvis_x = d.xpos[1, 0]
        tokens   = np.zeros((self.max_limbs, self.limb_obs_size), dtype=np.float32)

        for li in range(self.num_limbs):
            bi     = li + 1
            xpos   = d.xpos[bi].copy(); xpos[0] -= pelvis_x
            xvelp  = np.clip(d.cvel[bi, 3:6], -10, 10)
            xvelr  = d.cvel[bi, :3]
            expmap = quat_to_expmap(d.xquat[bi].copy())

            b = 0
            tokens[li, b:b+3] = xpos;   b += 3
            tokens[li, b:b+3] = xvelp;  b += 3
            tokens[li, b:b+3] = xvelr;  b += 3
            tokens[li, b:b+3] = expmap; b += 3
            tokens[li, b]     = 0. if li == 0 else angle_norm[li - 1]; b += 1
            if li == 0:
                tokens[li, b:b+2] = 0.
            else:
                tokens[li, b:b+2] = self.jrange_norm[li - 1]
            b += 2
            if li == 0:
                tokens[li, b:b+11] = global_state

        proprioceptive = tokens.reshape(self.max_limbs * self.limb_obs_size)

        dev = self.device
        obs = {
            "proprioceptive":   torch.tensor(
                proprioceptive, dtype=torch.float32, device=dev).unsqueeze(0),
            "context":          self.context,
            "edges":            self.edges,
            "traversals":       self.traversals,
            "SWAT_RE":          self.swat_re,
            "obs_padding_mask": self.obs_padding_mask,
            "act_padding_mask": self.act_padding_mask,
        }
        if self.graph_enc != "none":
            obs["graph_node_features"] = self.graph_node_features
            obs["graph_A_norm"]        = self.graph_A_norm

        return obs, phase_left, phase_right


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_pkg",  required=True,
                        help="Path to model_XXX_inference.pt")
    parser.add_argument("--xml_path",       required=True,
                        help="Stripped MuJoCo XML for sim + graph topology")
    parser.add_argument("--urdf_path",      default=None,
                        help="Variant URDF for 12-dim context only")
    parser.add_argument("--variant_name",   required=True,
                        help="Name of variant — if in training set uses exact stats, "
                             "otherwise uses OOD mean stats automatically")
    parser.add_argument("--output_file",    default="trajectory.pkl")
    parser.add_argument("--duration",       type=float, default=10.0)
    parser.add_argument("--cmd_vx",         type=float, default=0.5)
    parser.add_argument("--cmd_vy",         type=float, default=0.0)
    parser.add_argument("--cmd_yaw",        type=float, default=0.0)
    parser.add_argument("--render",         action="store_true", default=False)
    parser.add_argument("--device",         default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load inference package ────────────────────────────────────────────
    print(f"Loading inference package: {args.inference_pkg}")
    pkg = torch.load(args.inference_pkg, map_location=device)

    # ── Set cfg from package (no manual cfg setting needed) ──────────────
    cfg.MODEL.GRAPH_ENCODING       = pkg["graph_encoding"]
    cfg.MODEL.RWSE_K               = pkg["rwse_k"]
    cfg.MODEL.TRANSFORMER.USE_FILM = pkg["use_film"]
    cfg.MODEL.MAX_LIMBS            = pkg["max_limbs"]
    cfg.MODEL.MAX_JOINTS           = pkg["max_joints"]
    cfg.MODEL.GCN.HIDDEN_DIM       = pkg["gcn_hidden"]
    cfg.MODEL.GCN.OUT_DIM          = pkg["gcn_out"]
    cfg.MODEL.GCN.NUM_LAYERS       = pkg["gcn_layers"]
    cfg.PPO.NUM_ENVS               = 1
    cfg.PPO.BATCH_SIZE             = 1
    cfg.ENV.WALKERS                = ["g1"]

    # ── Normalization stats ───────────────────────────────────────────────
    # Seen variant → exact per-variant stats
    # OOD variant  → mean across all training variants (automatic)
    variant_names = pkg["variant_names"]
    if args.variant_name in variant_names:
        v_idx   = variant_names.index(args.variant_name)
        ob_mean = pkg["ob_mean_per_variant"][v_idx].to(device)
        ob_var  = pkg["ob_var_per_variant"][v_idx].to(device)
        print(f"Seen variant '{args.variant_name}' — using stats from row {v_idx}")
    else:
        ob_mean = pkg["ob_mean_ood"].to(device)
        ob_var  = pkg["ob_var_ood"].to(device)
        print(f"OOD variant '{args.variant_name}' — using mean stats "
              f"across {len(variant_names)} training variants")

    print(f"ob_mean: mean={ob_mean.mean():.3f}  std={ob_mean.std():.3f}")
    print(f"ob_var:  mean={ob_var.mean():.3f}")

    def normalize_obs(obs_dict):
        prop = obs_dict["proprioceptive"]
        prop = ((prop - ob_mean) / (ob_var + 1e-8).sqrt()).clamp(-10., 10.)
        obs_dict["proprioceptive"] = prop
        return obs_dict

    # ── Obs builder ───────────────────────────────────────────────────────
    obs_builder = DeployObsBuilder(
        args.xml_path, device, urdf_path=args.urdf_path)

    # ── Build observation space ───────────────────────────────────────────
    limb_obs_size = pkg["limb_obs_size"]
    prop_dim      = cfg.MODEL.MAX_LIMBS * limb_obs_size
    ctx_dim       = int(obs_builder.context.shape[1])
    inf           = float("inf")
    obs_spaces = {
        "proprioceptive":      _BoxSpace(-inf, inf, (prop_dim,)),
        "context":             _BoxSpace(-inf, inf, (ctx_dim,)),
        "obs_padding_mask":    _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS,)),
        "act_padding_mask":    _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS,)),
        "edges":               _BoxSpace(-inf, inf, (cfg.MODEL.MAX_JOINTS * 2,)),
        "traversals":          _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS,)),
        "SWAT_RE":             _BoxSpace(-inf, inf,
                                        (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS, 3)),
        "graph_node_features": _BoxSpace(-inf, inf,
                                        (cfg.MODEL.MAX_LIMBS, cfg.MODEL.RWSE_K)),
        "graph_A_norm":        _BoxSpace(0., 1.,
                                        (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS)),
    }
    observation_space = _DictSpace(obs_spaces)
    action_space      = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    # ── Load model ────────────────────────────────────────────────────────
    actor_critic = ActorCritic(observation_space, action_space).to(device)
    actor_critic.load_state_dict(pkg["model_state_dict"])
    actor_critic.eval()
    agent = Agent(actor_critic)
    print("Model loaded OK")
    print(f"  graph_enc={cfg.MODEL.GRAPH_ENCODING}  "
          f"film={cfg.MODEL.TRANSFORMER.USE_FILM}  "
          f"rwse_k={cfg.MODEL.RWSE_K}")

    # ── MuJoCo sim — always use XML ───────────────────────────────────────
    print(f"\n[Sim] Loading XML: {os.path.basename(args.xml_path)}")
    m = mujoco.MjModel.from_xml_path(args.xml_path)
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    print(f"  nq={m.nq}  nv={m.nv}  njnt={m.njnt}  nbody={m.nbody}")

    kp_arr         = get_kp_kd_arrays(m)[0]
    kd_arr         = get_kp_kd_arrays(m)[1]
    default_angles = get_default_angles(m)

    print(f"\nJoint order:")
    for i in range(12):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        print(f"  [{i}] {jn}  default={default_angles[i]:.3f}  "
              f"kp={kp_arr[i]}  kd={kd_arr[i]}")

    reset_sim(m, d, default_angles)

    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    print(f"\nVariant : {args.variant_name}")
    print(f"Command : vx={args.cmd_vx}  vy={args.cmd_vy}  yaw={args.cmd_yaw}")
    print(f"Duration: {args.duration}s")

    # ── Simulation loop ───────────────────────────────────────────────────
    max_sim_steps   = int(args.duration / SIM_DT)
    last_action     = np.zeros(12, dtype=np.float32)
    target_dof_pos  = default_angles.copy()
    episode_step    = 0
    sim_step        = 0
    trajectory      = []
    episode_rewards = []
    total_reward    = 0.0
    ep_len          = 0
    episodes        = 0
    t_start         = time.time()

    viewer_ctx = mujoco.viewer.launch_passive(m, d) if args.render else None

    try:
        while sim_step < max_sim_steps:

            # Physics
            tau       = ((target_dof_pos - d.qpos[7:]) * kp_arr +
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

            if viewer_ctx is not None:
                viewer_ctx.sync()

            if sim_step % DECIMATION != 0:
                continue

            # Build obs
            obs, phase_l, phase_r = obs_builder.build(
                d, command, episode_step, last_action)
            obs = normalize_obs(obs)

            # Policy
            with torch.no_grad():
                val, act, logp, dmv, dmmu = agent.act(obs, unimal_ids=[0])

            act_mask       = obs_builder.act_padding_mask[0].bool()
            real_action    = act[0, ~act_mask].cpu().numpy()
            target_dof_pos = real_action * ACTION_SCALE + default_angles
            last_action    = real_action.copy()
            episode_step  += 1

            # Reward
            height        = d.qpos[2]
            quat_wxyz     = d.qpos[3:7]
            base_vel_body = quat_rotate_inverse_np(quat_wxyz, d.qvel[:3])
            lin_err       = np.sum((command[:2] - base_vel_body[:2])**2)
            r = 0.15 * POLICY_DT + 1.0 * np.exp(-lin_err / 0.25) * POLICY_DT
            total_reward += r
            ep_len       += 1

            # Termination
            w, x, y, z = quat_wxyz
            roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            fell  = (height < 0.4 or height > 1.15
                     or abs(pitch) > 1.0 or abs(roll) > 0.8)

            if fell:
                ep_len_s = ep_len * POLICY_DT
                episode_rewards.append({
                    "episode":      episodes + 1,
                    "return":       round(total_reward, 3),
                    "length_steps": ep_len,
                    "length_secs":  round(ep_len_s, 2),
                    "termination":  "fall",
                })
                print(f"  Episode {episodes+1}: "
                      f"return={total_reward:.2f}  len={ep_len_s:.1f}s  FELL "
                      f"(h={height:.2f} p={pitch:.2f} r={roll:.2f})")
                reset_sim(m, d, default_angles)
                last_action    = np.zeros(12, dtype=np.float32)
                target_dof_pos = default_angles.copy()
                episode_step   = 0
                total_reward   = 0.0
                ep_len         = 0
                episodes      += 1

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    elapsed = time.time() - t_start
    print(f"\nDone: {sim_step} steps in {elapsed:.1f}s")

    if ep_len > 0:
        ep_len_s = ep_len * POLICY_DT
        episode_rewards.append({
            "episode":      episodes + 1,
            "return":       round(total_reward, 3),
            "length_steps": ep_len,
            "length_secs":  round(ep_len_s, 2),
            "termination":  "timeout",
        })
        print(f"  Episode {episodes+1} (final): "
              f"return={total_reward:.2f}  len={ep_len_s:.1f}s  timeout")

    print("\n── Episode Summary ──")
    if episode_rewards:
        returns = [e["return"]      for e in episode_rewards]
        lengths = [e["length_secs"] for e in episode_rewards]
        print(f"  Episodes:       {len(episode_rewards)}")
        print(f"  Mean return:    {np.mean(returns):.2f}")
        print(f"  Mean ep length: {np.mean(lengths):.1f}s")
        print(f"  Max return:     {np.max(returns):.2f}")
        print(f"  Max ep length:  {np.max(lengths):.1f}s")

    output = {
        "trajectory":      trajectory,
        "episode_rewards": episode_rewards,
        "variant_name":    args.variant_name,
        "command":         command.tolist(),
        "xml_path":        args.xml_path,
        "urdf_path":       args.urdf_path,
        "inference_pkg":   args.inference_pkg,
        "graph_encoding":  cfg.MODEL.GRAPH_ENCODING,
        "film":            cfg.MODEL.TRANSFORMER.USE_FILM,
        "sim_dt":          SIM_DT,
        "policy_dt":       POLICY_DT,
    }
    with open(args.output_file, "wb") as f:
        pickle.dump(output, f)
    print(f"\nTrajectory saved: {args.output_file}  ({len(trajectory)} steps)")


if __name__ == "__main__":
    main()