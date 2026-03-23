"""
deploy/deploy_mujoco/record_traj_modular.py

Records a trajectory using the trained transformer policy in MuJoCo.
Builds the same per-limb obs as training (no flat obs).
Loads ob_mean/ob_var from checkpoint for correct normalisation.

Usage:
    cd /home/sviswasam/dr/unitree_rl_gym
    python deploy/deploy_mujoco/record_traj_modular.py \
        --checkpoint output_walk_isaac_success/Mar21_18-51-07/model_400.pt \
        --xml_path /home/sviswasam/dr/ModuMorph/modular/unitree_g1_actual/xml/g1_12dof_stripped.xml \
        --output_file trajectory_g1.pkl \
        --duration 10.0 \
        --cmd_vx 0.5
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

# ── ModuMorph / modular_policy path ──────────────────────────────────────────
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic, Agent
from modular_policy.graphs.parser import MujocoGraphParser

# ── Constants ─────────────────────────────────────────────────────────────────
GAIT_PERIOD   = 0.8
GAIT_OFFSET   = 0.5
SIM_DT        = 0.005        # MuJoCo timestep
DECIMATION    = 4            # policy runs every 4 sim steps
POLICY_DT     = SIM_DT * DECIMATION   # = 0.02s

# PD gains — must match training
KP = {
    "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100,
    "knee": 150, "ankle": 40,
}
KD = {
    "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2,
    "knee": 4, "ankle": 2,
}
ACTION_SCALE  = 0.25

DEFAULT_ANGLES = {
    "left_hip_yaw_joint":     0.0,
    "left_hip_roll_joint":    0.0,
    "left_hip_pitch_joint":  -0.1,
    "left_knee_joint":        0.3,
    "left_ankle_pitch_joint":-0.2,
    "left_ankle_roll_joint":  0.0,
    "right_hip_yaw_joint":    0.0,
    "right_hip_roll_joint":   0.0,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint":       0.3,
    "right_ankle_pitch_joint":-0.2,
    "right_ankle_roll_joint": 0.0,
}


# ── Quaternion helpers ────────────────────────────────────────────────────────

def quat_wxyz_to_rot(q):
    """[w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quat_rotate_inverse_np(q, v):
    """Rotate v into body frame. q=[w,x,y,z]."""
    w, x, y, z = q
    t  = 2.0 * np.cross(np.array([x, y, z]), v)
    return v - w * t + np.cross(np.array([x, y, z]), t)


def mujoco_quat_to_wxyz(q):
    """MuJoCo stores [w,x,y,z] — already correct, just return as-is."""
    return q   # [w, x, y, z]


def quat_to_expmap(q):
    """[w,x,y,z] → axis-angle (expmap)."""
    w  = np.clip(q[0], -1 + 1e-7, 1 - 1e-7)
    xyz = q[1:]
    angle    = 2.0 * np.arccos(abs(w))
    sin_half = np.sin(angle / 2)
    if sin_half < 1e-8:
        return np.zeros(3)
    axis = xyz / sin_half
    return axis * angle


# ── Build per-limb obs (matches training exactly) ────────────────────────────

class MujocoObsBuilder:
    """
    Builds the same per-limb obs dict that the transformer expects,
    but from MuJoCo state instead of Isaac Gym tensors.
    """

    def __init__(self, xml_path, device):
        self.device    = device
        self.graph_enc = cfg.MODEL.GRAPH_ENCODING
        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        use_ltv = (self.graph_enc == "none")
        self.use_ltv       = use_ltv
        self.limb_obs_size = 32 if use_ltv else 26

        # Load MuJoCo model for static data
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        parser    = MujocoGraphParser(xml_path)
        self.num_limbs = parser.N
        num_pads  = self.max_limbs - self.num_limbs

        # ── Static tensors (same as ObsBuilder._load_graph_data) ─────────
        body_pos  = self.m.body_pos[1:self.num_limbs+1].copy()
        body_mass = self.m.body_mass[1:self.num_limbs+1].copy()
        bp_range  = body_pos.max(0) - body_pos.min(0) + 1e-8
        body_pos_n  = -1. + 2. * (body_pos - body_pos.min(0)) / bp_range
        bm_range    = body_mass.max() - body_mass.min() + 1e-8
        body_mass_n = -1. + 2. * (body_mass - body_mass.min()) / bm_range
        ctx     = np.concatenate([body_pos_n, body_mass_n[:, None]], axis=1)
        ctx_pad = np.zeros((self.max_limbs, ctx.shape[1]), dtype=np.float32)
        ctx_pad[:self.num_limbs] = ctx
        self.context = torch.tensor(
            ctx_pad.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

        jnt_range = self.m.jnt_range[1:13].copy()  # (12, 2)
        self.joint_lo   = jnt_range[:, 0].astype(np.float32)
        self.joint_hi   = jnt_range[:, 1].astype(np.float32)
        self.joint_span = np.clip(self.joint_hi - self.joint_lo, 1e-8, None)
        self.jrange_norm = np.stack(
            [self.joint_lo / math.pi, self.joint_hi / math.pi], axis=1)  # (12,2)

        # Edges
        joint_to   = self.m.jnt_bodyid[1:].copy() - 1
        parent_ids = self.m.body_parentid.copy()
        joint_from = np.array([parent_ids[c + 1] - 1 for c in joint_to])
        edges_raw  = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)
        edges_pad  = np.zeros(self.max_joints * 2, dtype=np.float32)
        edges_pad[:len(edges_raw)] = edges_raw
        self.edges = torch.tensor(edges_pad, device=device).unsqueeze(0)

        # Traversals
        parents = [-1] * self.num_limbs
        for i in range(len(joint_to)):
            ci, pi = int(joint_to[i]), int(joint_from[i])
            if 0 <= ci < self.num_limbs:
                parents[ci] = pi
        trav     = self._dfs_traversal(parents)
        trav_pad = np.zeros(self.max_limbs, dtype=np.float32)
        trav_pad[:len(trav)] = trav
        self.traversals = torch.tensor(trav_pad, device=device).unsqueeze(0)

        self.swat_re = torch.zeros(1, self.max_limbs, self.max_limbs, 3, device=device)

        obs_mask = [False]*self.num_limbs + [True]*num_pads
        act_mask = [True]  + [False]*(self.num_limbs-1) + [True]*num_pads
        self.obs_padding_mask = torch.tensor(obs_mask, dtype=torch.bool, device=device).unsqueeze(0)
        self.act_padding_mask = torch.tensor(act_mask, dtype=torch.bool, device=device).unsqueeze(0)

        if self.graph_enc != "none":
            feats  = parser.get_features(self.graph_enc)
            A_norm = parser.normalized_adjacency()
            N, fd  = feats.shape
            gf = torch.zeros(1, self.max_limbs, fd, device=device)
            gf[0, :N] = torch.tensor(feats, dtype=torch.float32, device=device)
            self.graph_node_features = gf
            ga = torch.zeros(1, self.max_limbs, self.max_limbs, device=device)
            ga[0, :N, :N] = torch.tensor(A_norm, dtype=torch.float32, device=device)
            self.graph_A_norm = ga
        else:
            self.graph_node_features = None
            self.graph_A_norm        = None

        # LTV for "none" encoding
        if use_ltv:
            ltvs = []
            for i in range(1, self.num_limbs + 1):
                bn = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i).lower()
                if   "hip_pitch" in bn: ltv = [1,0,0,0,0,0]
                elif "hip_roll"  in bn: ltv = [0,1,0,0,0,0]
                elif "hip_yaw"   in bn: ltv = [0,0,1,0,0,0]
                elif "knee"      in bn: ltv = [0,0,0,1,0,0]
                elif "ankle"     in bn: ltv = [0,0,0,0,1,0]
                else:                   ltv = [0,0,0,0,0,0]
                ltvs.append(ltv)
            self.ltv = np.array(ltvs, dtype=np.float32)
        else:
            self.ltv = None

        print(f"[MujocoObsBuilder] {self.num_limbs} limbs, "
              f"limb_obs_size={self.limb_obs_size}, graph_enc={self.graph_enc}")

    def _dfs_traversal(self, parents):
        root     = next(i for i, p in enumerate(parents) if p == -1)
        children = {i: [] for i in range(self.num_limbs)}
        for i, p in enumerate(parents):
            if p >= 0:
                children[p].append(i)
        order = []
        def dfs(n):
            order.append(n)
            for c in sorted(children[n]):
                dfs(c)
        dfs(root)
        return np.array(order, dtype=np.float32)

    def build(self, d, command, episode_step, last_action):
        """
        d:             mujoco.MjData
        command:       np.array [vx, vy, yaw]
        episode_step:  int
        last_action:   np.array (12,)

        Returns obs dict (batch=1) ready for ActorCritic.
        """
        # ── Root state ────────────────────────────────────────────────────
        quat_wxyz = mujoco_quat_to_wxyz(d.qpos[3:7])  # [w,x,y,z]
        ang_vel_w = d.qvel[3:6]
        base_ang_vel = quat_rotate_inverse_np(quat_wxyz, ang_vel_w) * 0.25

        grav_w   = np.array([0., 0., -1.])
        proj_grav = quat_rotate_inverse_np(quat_wxyz, grav_w)

        cmd_scaled = command * np.array([2.0, 2.0, 0.25])

        t           = episode_step * POLICY_DT
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        sin_ph      = np.sin(2. * math.pi * phase_left)
        cos_ph      = np.cos(2. * math.pi * phase_left)

        global_state = np.concatenate([
            base_ang_vel, proj_grav, cmd_scaled,
            [sin_ph], [cos_ph]])  # (11,)

        dof_pos = d.qpos[7:]   # (12,)
        angle_norm = (dof_pos - self.joint_lo) / self.joint_span

        # ── Per-body kinematics ───────────────────────────────────────────
        # xpos: body positions relative to world (or pelvis)
        # xvelp: body linear velocities
        # xvelr: body angular velocities
        pelvis_x = d.xpos[1, 0]   # body 1 = pelvis

        tokens = np.zeros((self.max_limbs, self.limb_obs_size), dtype=np.float32)

        for li in range(self.num_limbs):
            bi   = li + 1   # body index in MuJoCo (0=world)
            xpos = d.xpos[bi].copy()
            xpos[0] -= pelvis_x           # root-relative x

            xvelp = np.clip(d.cvel[bi, 3:6], -10, 10)  # linear vel
            xvelr = d.cvel[bi, :3]                       # angular vel

            # expmap from body quaternion
            body_quat = d.xquat[bi].copy()  # [w,x,y,z] in MuJoCo
            expmap = quat_to_expmap(body_quat)

            b = 0
            tokens[li, b:b+3] = xpos;   b += 3
            tokens[li, b:b+3] = xvelp;  b += 3
            tokens[li, b:b+3] = xvelr;  b += 3
            tokens[li, b:b+3] = expmap; b += 3

            if self.use_ltv:
                tokens[li, b:b+6] = self.ltv[li]; b += 6

            tokens[li, b] = 0. if li == 0 else angle_norm[li - 1]
            b += 1

            if li == 0:
                tokens[li, b:b+2] = 0.
            else:
                tokens[li, b:b+2] = self.jrange_norm[li - 1]
            b += 2

            if li == 0:
                tokens[li, b:b+11] = global_state

        proprioceptive = tokens.reshape(self.max_limbs * self.limb_obs_size)

        # ── Build obs dict as tensors (batch=1) ───────────────────────────
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


# ── PD control ────────────────────────────────────────────────────────────────

def get_kp_kd_arrays(m):
    """Build kp, kd arrays in joint order from MuJoCo model."""
    kp_arr = np.zeros(12, dtype=np.float32)
    kd_arr = np.zeros(12, dtype=np.float32)
    for i in range(12):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1).lower()
        for key in KP:
            if key in jname:
                kp_arr[i] = KP[key]
                kd_arr[i] = KD[key]
                break
    return kp_arr, kd_arr


def get_default_angles(m):
    """Build default angle array in joint order from MuJoCo model."""
    default = np.zeros(12, dtype=np.float32)
    for i in range(12):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        if jname in DEFAULT_ANGLES:
            default[i] = DEFAULT_ANGLES[jname]
    return default


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model_*.pt checkpoint")
    parser.add_argument("--xml_path",   required=True,
                        help="MuJoCo XML path (stripped variant)")
    parser.add_argument("--output_file", default="trajectory.pkl",
                        help="Output .pkl path")
    parser.add_argument("--duration",   type=float, default=10.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--cmd_vx",     type=float, default=0.5)
    parser.add_argument("--cmd_vy",     type=float, default=0.0)
    parser.add_argument("--cmd_yaw",    type=float, default=0.0)
    parser.add_argument("--render",     action="store_true", default=False,
                        help="Open MuJoCo viewer (requires display)")
    parser.add_argument("--graph_encoding", default="topological",
                        choices=["none", "onehot", "topological"])
    parser.add_argument("--device",     default="cpu",
                        help="cpu or cuda:0 (cpu fine for single env inference)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Config ────────────────────────────────────────────────────────────
    cfg.MODEL.GRAPH_ENCODING = args.graph_encoding
    cfg.MODEL.MAX_LIMBS      = 13
    cfg.MODEL.MAX_JOINTS     = 12
    cfg.PPO.NUM_ENVS         = 1
    cfg.PPO.BATCH_SIZE       = 1
    cfg.ENV.WALKERS          = ["g1"]

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # ── Obs builder ───────────────────────────────────────────────────────
    obs_builder = MujocoObsBuilder(args.xml_path, device)

    # ── Build ActorCritic and load weights ────────────────────────────────
    # Build spaces manually since MujocoObsBuilder doesn't have them yet
    from modular_policy.algos.ppo.obs_builder import _BoxSpace, _DictSpace
    prop_dim = cfg.MODEL.MAX_LIMBS * obs_builder.limb_obs_size
    ctx_dim  = int(obs_builder.context.shape[1])
    inf      = float("inf")
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
        obs_spaces["graph_node_features"] = _BoxSpace(-inf, inf, (cfg.MODEL.MAX_LIMBS, fd))
        obs_spaces["graph_A_norm"]        = _BoxSpace(0., 1.,
                                                      (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS))
    observation_space = _DictSpace(obs_spaces)
    action_space      = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    actor_critic = ActorCritic(observation_space, action_space).to(device)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()
    agent = Agent(actor_critic)

    # ── Obs normalisation stats ───────────────────────────────────────────
    ob_mean  = ckpt.get("ob_mean", torch.zeros(prop_dim)).to(device)
    ob_var   = ckpt.get("ob_var",  torch.ones(prop_dim)).to(device)
    clipob   = 10.0

    def normalize_obs(obs_dict):
        prop = obs_dict["proprioceptive"]
        prop = ((prop - ob_mean) / (ob_var + 1e-8).sqrt()).clamp(-clipob, clipob)
        obs_dict["proprioceptive"] = prop
        return obs_dict

    # ── MuJoCo setup ─────────────────────────────────────────────────────
    m = obs_builder.m
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)

    kp_arr, kd_arr = get_kp_kd_arrays(m)
    default_angles = get_default_angles(m)

    print(f"Joint order:")
    for i in range(12):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        print(f"  [{i}] {jn}  default={default_angles[i]:.3f}  "
              f"kp={kp_arr[i]}  kd={kd_arr[i]}")

    # Reset to default pose
    mujoco.mj_resetData(m, d)
    d.qpos[2]  = 0.8   # spawn height
    d.qpos[3]  = 1.0   # quat w
    d.qpos[7:] = default_angles
    mujoco.mj_forward(m, d)

    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    print(f"\nCommand: vx={args.cmd_vx}  vy={args.cmd_vy}  yaw={args.cmd_yaw}")
    print(f"Duration: {args.duration}s  ({int(args.duration / SIM_DT)} sim steps)")
    print(f"Graph encoding: {args.graph_encoding}")

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

    t_start = time.time()

    if args.render:
        viewer_ctx = mujoco.viewer.launch_passive(m, d)
    else:
        viewer_ctx = None

    try:
        while sim_step < max_sim_steps:

            # ── Physics sub-step ──────────────────────────────────────────
            tau = (target_dof_pos - d.qpos[7:]) * kp_arr + \
                  (np.zeros(12)   - d.qvel[6:]) * kd_arr
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            sim_step += 1

            # ── Record state ──────────────────────────────────────────────
            trajectory.append({
                "qpos":        d.qpos.copy(),
                "qvel":        d.qvel.copy(),
                "action":      last_action.copy(),
                "target_pos":  target_dof_pos.copy(),
                "time":        d.time,
                "episode_step": episode_step,
            })

            if viewer_ctx is not None:
                viewer_ctx.sync()

            # ── Policy step (every DECIMATION sim steps) ──────────────────
            if sim_step % DECIMATION != 0:
                continue

            # Build obs
            obs, phase_l, phase_r = obs_builder.build(
                d, command, episode_step, last_action)
            obs = normalize_obs(obs)

            # Inference
            with torch.no_grad():
                val, act, logp, dmv, dmmu = agent.act(
                    obs, unimal_ids=[0])

            # Strip padding → real 12 actions
            act_mask    = obs_builder.act_padding_mask[0].bool()
            real_action = act[0, ~act_mask].cpu().numpy()  # (12,)

            target_dof_pos = real_action * ACTION_SCALE + default_angles
            last_action    = real_action.copy()
            episode_step  += 1

            # ── Simple reward tracking ────────────────────────────────────
            height = d.qpos[2]
            # alive reward proxy
            r = 0.15 * POLICY_DT
            # tracking reward proxy
            base_vel_world = d.qvel[:3]
            quat = d.qpos[3:7]
            base_vel_body  = quat_rotate_inverse_np(quat, base_vel_world)
            lin_err = np.sum((command[:2] - base_vel_body[:2])**2)
            r += 1.0 * np.exp(-lin_err / 0.25) * POLICY_DT
            total_reward += r
            ep_len       += 1

            # ── Termination check ─────────────────────────────────────────
            # Pelvis contact or height out of range
            pelvis_contact = np.linalg.norm(
                d.cfrc_ext[1, 3:6]) > 1.0 if d.ncon > 0 else False
            height_fail = height < 0.4 or height > 1.15

            # Roll/pitch from quat
            w, x, y, z = quat
            roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            angle_fail = abs(pitch) > 1.0 or abs(roll) > 0.8

            if pelvis_contact or height_fail or angle_fail:
                ep_len_s = ep_len * POLICY_DT
                episode_rewards.append({
                    "episode": episodes + 1,
                    "return":  round(total_reward, 3),
                    "length_steps": ep_len,
                    "length_secs":  round(ep_len_s, 2),
                    "termination":  "fall" if (height_fail or angle_fail)
                                    else "contact",
                })
                print(f"  Episode {episodes+1}: "
                      f"return={total_reward:.2f}  "
                      f"len={ep_len_s:.1f}s  "
                      f"{'FELL' if height_fail or angle_fail else 'contact'}")

                # Reset
                mujoco.mj_resetData(m, d)
                d.qpos[2]  = 0.8
                d.qpos[3]  = 1.0
                d.qpos[7:] = default_angles
                mujoco.mj_forward(m, d)
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
    print(f"\nSimulation complete: {sim_step} sim steps in {elapsed:.1f}s "
          f"({sim_step/elapsed:.0f} steps/s)")

    # Final episode stats
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

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Episode Summary ──")
    if episode_rewards:
        returns = [e["return"]       for e in episode_rewards]
        lengths = [e["length_secs"]  for e in episode_rewards]
        print(f"  Episodes:        {len(episode_rewards)}")
        print(f"  Mean return:     {np.mean(returns):.2f}")
        print(f"  Mean ep length:  {np.mean(lengths):.1f}s")
        print(f"  Max ep length:   {np.max(lengths):.1f}s")
        print(f"  Max return:      {np.max(returns):.2f}")

    # ── Save trajectory ───────────────────────────────────────────────────
    output = {
        "trajectory":      trajectory,
        "episode_rewards": episode_rewards,
        "command":         command.tolist(),
        "xml_path":        args.xml_path,
        "checkpoint":      args.checkpoint,
        "graph_encoding":  args.graph_encoding,
        "sim_dt":          SIM_DT,
        "policy_dt":       POLICY_DT,
    }
    with open(args.output_file, "wb") as f:
        pickle.dump(output, f)
    print(f"\nTrajectory saved to: {args.output_file}")
    print(f"  Steps: {len(trajectory)}")


if __name__ == "__main__":
    main()