"""
modular_policy/algos/ppo/obs_builder.py

Builds per-limb observation tokens from Isaac Gym raw state tensors.
No gym dependency — uses plain _BoxSpace / _DictSpace instead of gym.spaces.

Per-limb token layout:
  graph_encoding == "none":  xpos(3)+xvelp(3)+xvelr(3)+expmap(3)+ltv(6)+angle(1)+jrange(2)+global(11) = 32
  graph_encoding != "none":  xpos(3)+xvelp(3)+xvelr(3)+expmap(3)+angle(1)+jrange(2)+global(11)        = 26

global(11) on root body token only, zeros elsewhere.
"""

import math
import os
import numpy as np
import torch

from modular_policy.config import cfg
from modular_policy.graphs.parser import MujocoGraphParser

GAIT_PERIOD = 0.8
GAIT_OFFSET = 0.5


# ── Minimal space objects (replace gym.spaces) ───────────────────────────────

class _BoxSpace:
    """Minimal Box space — holds shape and dtype, no gym needed."""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low   = low
        self.high  = high
        self.shape = shape
        self.dtype = dtype


class _DictSpace:
    """Minimal Dict space — holds a dict of _BoxSpace objects."""
    def __init__(self, spaces_dict):
        self.spaces = spaces_dict   # dict[str, _BoxSpace]

    def __getitem__(self, key):
        return self.spaces[key]

    def keys(self):
        return self.spaces.keys()

    def items(self):
        return self.spaces.items()


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _ig_quat_to_wxyz(q):
    """Isaac Gym [x,y,z,w] → [w,x,y,z]."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _quat_rotate_inverse(q, v):
    """q: (...,4)[w,x,y,z], v: (...,3) → body-frame v."""
    w   = q[..., 0:1]
    xyz = q[..., 1:]
    t   = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def _batch_quat_to_expmap(quat):
    """quat: (...,4)[w,x,y,z] → expmap (...,3)."""
    w        = quat[..., 0].clamp(-1 + 1e-7, 1 - 1e-7)
    xyz      = quat[..., 1:]
    angle    = 2.0 * torch.acos(w.abs())
    sin_half = torch.sin(angle / 2).clamp(min=1e-8)
    axis     = xyz / sin_half.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


# ── Main class ────────────────────────────────────────────────────────────────

class ObsBuilder:
    """
    Call build() every policy step to get the obs dict.
    Exposes observation_space (_DictSpace) and action_space (_BoxSpace)
    for Buffer and ActorCritic.
    """

    def __init__(self, env, xml_path, device):
        self.env         = env
        self.device      = device
        self.num_envs    = env.num_envs
        self.num_actions = env.num_actions   # 12
        self.max_limbs   = cfg.MODEL.MAX_LIMBS   # 13
        self.max_joints  = cfg.MODEL.MAX_JOINTS  # 12
        self.graph_enc   = cfg.MODEL.GRAPH_ENCODING

        use_ltv = (self.graph_enc == "none")
        self.use_ltv       = use_ltv
        self.limb_obs_size = 32 if use_ltv else 26

        # Static graph data from XML
        self._load_graph_data(xml_path)

        # Joint ranges from Isaac Gym
        dof_props = env.gym.get_actor_dof_properties(
            env.envs[0], env.actor_handles[0])
        self.joint_lo   = torch.tensor(
            [float(dof_props["lower"][i]) for i in range(self.num_actions)],
            dtype=torch.float32, device=device)
        self.joint_hi   = torch.tensor(
            [float(dof_props["upper"][i]) for i in range(self.num_actions)],
            dtype=torch.float32, device=device)
        self.joint_span = (self.joint_hi - self.joint_lo).clamp(min=1e-8)
        self.jrange_norm = torch.stack(
            [self.joint_lo / math.pi, self.joint_hi / math.pi], dim=1)  # (12,2)

        # Limb-type one-hot for "none" encoding
        if use_ltv:
            bnames = env.gym.get_actor_rigid_body_names(
                env.envs[0], env.actor_handles[0])
            ltvs = []
            for bn in bnames:
                n = bn.lower()
                if   "hip_pitch" in n: ltv = [1,0,0,0,0,0]
                elif "hip_roll"  in n: ltv = [0,1,0,0,0,0]
                elif "hip_yaw"   in n: ltv = [0,0,1,0,0,0]
                elif "knee"      in n: ltv = [0,0,0,1,0,0]
                elif "ankle"     in n: ltv = [0,0,0,0,1,0]
                else:                  ltv = [0,0,0,0,0,0]
                ltvs.append(ltv)
            self.ltv = torch.tensor(ltvs, dtype=torch.float32, device=device)

        # Foot indices for reward computation
        bnames = env.gym.get_actor_rigid_body_names(
            env.envs[0], env.actor_handles[0])
        self.left_foot_idx  = next(
            i for i, n in enumerate(bnames) if "left_ankle_roll"  in n.lower())
        self.right_foot_idx = next(
            i for i, n in enumerate(bnames) if "right_ankle_roll" in n.lower())
        self.feet_indices = torch.tensor(
            [self.left_foot_idx, self.right_foot_idx],
            dtype=torch.long, device=device)

        # Build spaces
        prop_dim = self.max_limbs * self.limb_obs_size
        ctx_dim  = int(self.context.shape[1])
        inf      = float("inf")

        obs_spaces = {
            "proprioceptive":   _BoxSpace(-inf, inf, (prop_dim,)),
            "context":          _BoxSpace(-inf, inf, (ctx_dim,)),
            "obs_padding_mask": _BoxSpace(-inf, inf, (self.max_limbs,)),
            "act_padding_mask": _BoxSpace(-inf, inf, (self.max_limbs,)),
            "edges":            _BoxSpace(-inf, inf, (self.max_joints * 2,)),
            "traversals":       _BoxSpace(-inf, inf, (self.max_limbs,)),
            "SWAT_RE":          _BoxSpace(-inf, inf, (self.max_limbs, self.max_limbs, 3)),
        }
        if self.graph_enc != "none":
            feat_dim = 7 if self.graph_enc == "onehot" else 6
            obs_spaces["graph_node_features"] = _BoxSpace(
                -inf, inf, (self.max_limbs, feat_dim))
            obs_spaces["graph_A_norm"] = _BoxSpace(
                0., 1., (self.max_limbs, self.max_limbs))

        self.observation_space = _DictSpace(obs_spaces)
        self.action_space      = _BoxSpace(-1., 1., (self.max_limbs,))

        print(f"[ObsBuilder] limb_obs_size={self.limb_obs_size} "
              f"prop_dim={prop_dim} graph_enc={self.graph_enc}")

    # ── Static graph data ─────────────────────────────────────────────────────

    def _load_graph_data(self, xml_path):
        """Read graph structure from MuJoCo XML using mujoco3 bindings."""
        print("[ObsBuilder] Loading graph data...", flush=True)
        import mujoco
        print("[ObsBuilder] mujoco imported", flush=True)
        m = mujoco.MjModel.from_xml_path(xml_path)
        print("[ObsBuilder] XML loaded", flush=True)

        parser    = MujocoGraphParser(xml_path)
        num_limbs = parser.N
        num_pads  = self.max_limbs - num_limbs

        # Context: per-body position(3) + mass(1), normalised to [-1,1]
        body_pos  = m.body_pos[1:num_limbs+1].copy()
        body_mass = m.body_mass[1:num_limbs+1].copy()
        bp_range  = body_pos.max(0) - body_pos.min(0) + 1e-8
        body_pos_n  = -1. + 2. * (body_pos - body_pos.min(0)) / bp_range
        bm_range    = body_mass.max() - body_mass.min() + 1e-8
        body_mass_n = -1. + 2. * (body_mass - body_mass.min()) / bm_range
        ctx = np.concatenate([body_pos_n, body_mass_n[:, None]], axis=1)  # (L,4)
        ctx_pad = np.zeros((self.max_limbs, ctx.shape[1]), dtype=np.float32)
        ctx_pad[:num_limbs] = ctx
        self.context = torch.tensor(
            ctx_pad.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)

        # Edges
        joint_to   = m.jnt_bodyid[1:].copy() - 1
        parent_ids = m.body_parentid.copy()
        joint_from = np.array([parent_ids[c + 1] - 1 for c in joint_to])
        edges_raw  = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)
        edges_pad  = np.zeros(self.max_joints * 2, dtype=np.float32)
        edges_pad[:len(edges_raw)] = edges_raw
        self.edges = torch.tensor(
            edges_pad, device=self.device).unsqueeze(0)

        # Traversal
        parents = [-1] * num_limbs
        for i in range(len(joint_to)):
            ci, pi = int(joint_to[i]), int(joint_from[i])
            if 0 <= ci < num_limbs:
                parents[ci] = pi
        trav     = self._dfs_traversal(parents, num_limbs)
        trav_pad = np.zeros(self.max_limbs, dtype=np.float32)
        trav_pad[:len(trav)] = trav
        self.traversals = torch.tensor(
            trav_pad, device=self.device).unsqueeze(0)

        # SWAT_RE (zeros — not used in default config)
        self.swat_re = torch.zeros(
            1, self.max_limbs, self.max_limbs, 3, device=self.device)

        # Padding masks
        obs_mask = [False]*num_limbs + [True]*num_pads
        act_mask = [True]  + [False]*(num_limbs-1) + [True]*num_pads
        self.obs_padding_mask = torch.tensor(
            obs_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        self.act_padding_mask = torch.tensor(
            act_mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        # GCN features
        if self.graph_enc != "none":
            feats  = parser.get_features(self.graph_enc)
            A_norm = parser.normalized_adjacency()
            N, fd  = feats.shape
            gf = torch.zeros(1, self.max_limbs, fd, device=self.device)
            gf[0, :N] = torch.tensor(feats, dtype=torch.float32, device=self.device)
            self.graph_node_features = gf
            ga = torch.zeros(1, self.max_limbs, self.max_limbs, device=self.device)
            ga[0, :N, :N] = torch.tensor(A_norm, dtype=torch.float32, device=self.device)
            self.graph_A_norm = ga
        else:
            self.graph_node_features = None
            self.graph_A_norm        = None

        self._num_limbs = num_limbs
        print(f"[ObsBuilder] Graph: {num_limbs} limbs from {os.path.basename(xml_path)}")

    @staticmethod
    def _dfs_traversal(parents, num_limbs):
        root     = next(i for i, p in enumerate(parents) if p == -1)
        children = {i: [] for i in range(num_limbs)}
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

    # ── Per-step build ────────────────────────────────────────────────────────

    def build(self, commands, episode_steps, last_actions, dt):
        """
        commands:      (N, 3)
        episode_steps: (N,) int
        last_actions:  (N, 12)
        dt:            scalar
        Returns (obs_dict, phase_left, phase_right)
        """
        N   = self.num_envs
        env = self.env

        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        root_states = env.root_states[:N]
        dof_pos     = env.dof_pos[:N]
        dof_vel     = env.dof_vel[:N]
        rigid_body  = env.rigid_body_states_view[:N]

        quat_wxyz = _ig_quat_to_wxyz(root_states[:, 3:7])

        # Global state for root token
        ang_vel_w    = root_states[:, 10:13]
        base_ang_vel = _quat_rotate_inverse(quat_wxyz, ang_vel_w) * 0.25

        grav_w       = torch.tensor(
            [0., 0., -1.], device=self.device).expand(N, 3)
        proj_gravity = _quat_rotate_inverse(quat_wxyz, grav_w)

        cmd_scaled = commands * torch.tensor(
            [2.0, 2.0, 0.25], device=self.device)

        t           = episode_steps.float() * dt
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        sin_ph      = torch.sin(2. * math.pi * phase_left).unsqueeze(1)
        cos_ph      = torch.cos(2. * math.pi * phase_left).unsqueeze(1)

        global_state = torch.cat(
            [base_ang_vel, proj_gravity, cmd_scaled, sin_ph, cos_ph], dim=1)  # (N,11)

        # Per-body kinematics
        num_limbs = self._num_limbs
        body_pos  = rigid_body[:, :num_limbs, 0:3]
        body_velp = rigid_body[:, :num_limbs, 7:10]
        body_velr = rigid_body[:, :num_limbs, 10:13]
        body_quat = _ig_quat_to_wxyz(rigid_body[:, :num_limbs, 3:7])

        root_x = body_pos[:, 0:1, 0:1]
        xpos   = body_pos.clone(); xpos[:, :, 0:1] -= root_x
        xvelp  = body_velp.clamp(-10, 10)
        xvelr  = body_velr
        expmap = _batch_quat_to_expmap(body_quat)

        angle_norm = (dof_pos - self.joint_lo) / self.joint_span
        jrange     = self.jrange_norm.unsqueeze(0).expand(N, -1, -1)

        # Assemble tokens
        tokens = torch.zeros(
            N, self.max_limbs, self.limb_obs_size, device=self.device)

        for li in range(num_limbs):
            b = 0
            tokens[:, li, b:b+3] = xpos[:, li];   b += 3
            tokens[:, li, b:b+3] = xvelp[:, li];  b += 3
            tokens[:, li, b:b+3] = xvelr[:, li];  b += 3
            tokens[:, li, b:b+3] = expmap[:, li]; b += 3

            if self.use_ltv:
                tokens[:, li, b:b+6] = self.ltv[li]; b += 6

            tokens[:, li, b] = 0. if li == 0 else angle_norm[:, li - 1]
            b += 1

            if li == 0:
                tokens[:, li, b:b+2] = 0.
            else:
                tokens[:, li, b:b+2] = jrange[:, li - 1]
            b += 2

            if li == 0:
                tokens[:, li, b:b+11] = global_state

        proprioceptive = tokens.reshape(N, self.max_limbs * self.limb_obs_size)

        obs = {
            "proprioceptive":   proprioceptive,
            "context":          self.context.expand(N, -1),
            "edges":            self.edges.expand(N, -1),
            "traversals":       self.traversals.expand(N, -1),
            "SWAT_RE":          self.swat_re.expand(N, -1, -1, -1),
            "obs_padding_mask": self.obs_padding_mask.expand(N, -1),
            "act_padding_mask": self.act_padding_mask.expand(N, -1),
        }
        if self.graph_enc != "none":
            obs["graph_node_features"] = self.graph_node_features.expand(N, -1, -1)
            obs["graph_A_norm"]        = self.graph_A_norm.expand(N, -1, -1)

        return obs, phase_left, phase_right