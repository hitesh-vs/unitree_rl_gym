"""
modular_policy/algos/ppo/obs_builder.py

Builds per-limb observation tokens from Isaac Gym raw state tensors.
Supports multi-variant training: loads graph data for each variant XML,
selects the correct context/edges/graph features per env at build time.

Two-path loading:
  xml_path  — stripped MuJoCo XML for graph topology (worldbody, adjacency, RWSE)
  urdf_path — variant URDF for actual kinematics/mass (body positions, masses, joint ranges)

No gym dependency — uses plain _BoxSpace / _DictSpace.
"""

import math
import os
import numpy as np
import torch

from modular_policy.config import cfg
from modular_policy.graphs.parser import MujocoGraphParser

GAIT_PERIOD = 0.8
GAIT_OFFSET = 0.5


# ── Minimal space objects ─────────────────────────────────────────────────────

class _BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low   = low
        self.high  = high
        self.shape = shape
        self.dtype = dtype


class _DictSpace:
    def __init__(self, spaces_dict):
        self.spaces = spaces_dict

    def __getitem__(self, key):
        return self.spaces[key]

    def keys(self):
        return self.spaces.keys()

    def items(self):
        return self.spaces.items()


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _ig_quat_to_wxyz(q):
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _quat_rotate_inverse(q, v):
    w   = q[..., 0:1]
    xyz = q[..., 1:]
    t   = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def _batch_quat_to_expmap(quat):
    w        = quat[..., 0].clamp(-1 + 1e-7, 1 - 1e-7)
    xyz      = quat[..., 1:]
    angle    = 2.0 * torch.acos(w.abs())
    sin_half = torch.sin(angle / 2).clamp(min=1e-8)
    axis     = xyz / sin_half.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


# ── Graph data for one variant ────────────────────────────────────────────────

class _VariantGraphData:
    """Holds static graph tensors for one robot variant."""
    pass


def _load_variant_graph(xml_path, max_limbs, max_joints,
                         graph_enc, use_ltv, num_actions,
                         device, urdf_path=None):
    """
    Load graph data for one variant.

    xml_path:  stripped MuJoCo XML — used for graph topology (worldbody,
               adjacency matrix, RWSE features). Must have <worldbody>.
    urdf_path: variant URDF — used for body positions, masses, joint ranges.
               If None, falls back to xml_path for everything.

    Returns a _VariantGraphData object.
    """
    import mujoco

    # ── Load both models up front ─────────────────────────────────────────
    # m_topo: graph topology from stripped XML (has <worldbody>, 14 bodies)
    # m:      kinematics/mass from variant URDF (13 bodies, pelvis=world)
    m_topo   = mujoco.MjModel.from_xml_path(xml_path)
    kin_path = urdf_path if urdf_path else xml_path
    m        = mujoco.MjModel.from_xml_path(kin_path)

    # At top of _load_variant_graph, after loading models:
    print(f"  [load] xml={os.path.basename(xml_path)} "
          f"kin={os.path.basename(kin_path)} "
          f"topo_nbody={m_topo.nbody} kin_nbody={m.nbody}", flush=True)

    # ── Graph topology from stripped XML ──────────────────────────────────
    parser    = MujocoGraphParser(xml_path)
    num_limbs = parser.N
    num_pads  = max_limbs - num_limbs

    # ── Kinematics/mass from URDF (or XML if no URDF given) ───────────────
    kin_path = urdf_path if urdf_path else xml_path
    m        = mujoco.MjModel.from_xml_path(kin_path)

    vd = _VariantGraphData()
    vd.num_limbs = num_limbs

    try:
        # ── Context ───────────────────────────────────────────────────────
        topo_names   = [m_topo.body(i).name for i in range(1, m_topo.nbody)]
        kin_names    = [m.body(i).name      for i in range(1, m.nbody)]
        kin_name_set = set(kin_names)

        body_pos_list  = []
        body_mass_list = []
        for bname in topo_names:
            if bname in kin_name_set:
                idx = kin_names.index(bname) + 1
                body_pos_list.append(m.body_pos[idx].copy())
                body_mass_list.append(float(m.body_mass[idx]))
            else:
                topo_idx = topo_names.index(bname) + 1
                body_pos_list.append(m_topo.body_pos[topo_idx].copy())
                body_mass_list.append(float(m_topo.body_mass[topo_idx]))

        body_pos  = np.array(body_pos_list,  dtype=np.float32)
        body_mass = np.array(body_mass_list, dtype=np.float32)

        bp_range    = body_pos.max(0) - body_pos.min(0) + 1e-8
        body_pos_n  = -1. + 2. * (body_pos - body_pos.min(0)) / bp_range
        bm_range    = body_mass.max() - body_mass.min() + 1e-8
        body_mass_n = -1. + 2. * (body_mass - body_mass.min()) / bm_range
        ctx     = np.concatenate([body_pos_n, body_mass_n[:, None]], axis=1)
        ctx_pad = np.zeros((max_limbs, ctx.shape[1]), dtype=np.float32)
        ctx_pad[:num_limbs] = ctx
        vd.context = torch.tensor(
            ctx_pad.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        print(f"    context OK: shape={vd.context.shape}", flush=True)

    except Exception as e:
        print(f"  ERROR in context block: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise

    # ── Edges from topology XML ───────────────────────────────────────────
    # Use the topology model for structural edges
    m_topo     = mujoco.MjModel.from_xml_path(xml_path)
    joint_to   = m_topo.jnt_bodyid[1:].copy() - 1
    parent_ids = m_topo.body_parentid.copy()
    joint_from = np.array([parent_ids[c + 1] - 1 for c in joint_to])
    edges_raw  = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)
    edges_pad  = np.zeros(max_joints * 2, dtype=np.float32)
    edges_pad[:len(edges_raw)] = edges_raw
    vd.edges = torch.tensor(edges_pad, device=device).unsqueeze(0)

    # ── Traversals ────────────────────────────────────────────────────────
    parents = [-1] * num_limbs
    for i in range(len(joint_to)):
        ci, pi = int(joint_to[i]), int(joint_from[i])
        if 0 <= ci < num_limbs:
            parents[ci] = pi
    trav     = _dfs_traversal(parents, num_limbs)
    trav_pad = np.zeros(max_limbs, dtype=np.float32)
    trav_pad[:len(trav)] = trav
    vd.traversals = torch.tensor(trav_pad, device=device).unsqueeze(0)

    # ── SWAT_RE ───────────────────────────────────────────────────────────
    vd.swat_re = torch.zeros(1, max_limbs, max_limbs, 3, device=device)

    # ── Padding masks ─────────────────────────────────────────────────────
    obs_mask = [False]*num_limbs + [True]*num_pads
    act_mask = [True]  + [False]*(num_limbs-1) + [True]*num_pads
    vd.obs_padding_mask = torch.tensor(
        obs_mask, dtype=torch.bool, device=device).unsqueeze(0)
    vd.act_padding_mask = torch.tensor(
        act_mask, dtype=torch.bool, device=device).unsqueeze(0)

    # ── GCN features from topology XML ────────────────────────────────────
    if graph_enc != "none":
        feats  = parser.get_features(graph_enc)
        A_norm = parser.normalized_adjacency()
        N, fd  = feats.shape
        gf = torch.zeros(1, max_limbs, fd, device=device)
        gf[0, :N] = torch.tensor(feats, dtype=torch.float32, device=device)
        vd.graph_node_features = gf
        ga = torch.zeros(1, max_limbs, max_limbs, device=device)
        ga[0, :N, :N] = torch.tensor(A_norm, dtype=torch.float32, device=device)
        vd.graph_A_norm = ga
    else:
        vd.graph_node_features = None
        vd.graph_A_norm        = None

    # ── Joint ranges from variant kinematics ──────────────────────────────
    n_jnts    = m.njnt - 1   # exclude free joint = 11 for URDF
    jnt_range = m.jnt_range[1:n_jnts+1].copy()        # (11, 2)

    # Pad to num_actions (12) using symmetric range from topology XML
    if n_jnts < num_actions:
        topo_jnt_range = m_topo.jnt_range[1:num_actions+1].copy()  # (12, 2)
        # Fill missing joints with topo ranges (base robot values)
        full_range = topo_jnt_range.copy()
        full_range[:n_jnts] = jnt_range
        jnt_range  = full_range
        print(f"  Padded joint ranges from {n_jnts} to {num_actions}", flush=True)
    lo = torch.tensor(jnt_range[:, 0], dtype=torch.float32, device=device)
    hi = torch.tensor(jnt_range[:, 1], dtype=torch.float32, device=device)
    vd.joint_lo    = lo
    vd.joint_hi    = hi
    vd.joint_span  = (hi - lo).clamp(min=1e-8)
    vd.jrange_norm = torch.stack([lo / math.pi, hi / math.pi], dim=1)  # (12,2)

    # ── Limb-type one-hot ─────────────────────────────────────────────────
    if use_ltv:
        bnames = [m.body(i).name for i in range(1, num_limbs+1)]
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
        vd.ltv = torch.tensor(ltvs, dtype=torch.float32, device=device)
    else:
        vd.ltv = None

    return vd


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


# ── Main ObsBuilder ───────────────────────────────────────────────────────────

class ObsBuilder:
    """
    Supports both single-variant and multi-variant training.

    Single variant: pass xml_path (str)
    Multi-variant:  pass xml_paths (topology, same for all) +
                         urdf_paths (kinematics, different per variant) +
                         env_variant_ids tensor
    """

    def __init__(self, env, xml_path, device,
                 xml_paths=None, urdf_paths=None, env_variant_ids=None):
        """
        env:              Isaac Gym env
        xml_path:         str — base stripped XML (always required)
        device:           str
        xml_paths:        list[str] — topology XMLs, one per variant
        urdf_paths:       list[str] — variant URDFs for kinematics/mass
        env_variant_ids:  (N,) long tensor
        """
        self.env         = env
        self.device      = device
        self.num_envs    = env.num_envs
        self.num_actions = env.num_actions
        self.max_limbs   = cfg.MODEL.MAX_LIMBS
        self.max_joints  = cfg.MODEL.MAX_JOINTS
        self.graph_enc   = cfg.MODEL.GRAPH_ENCODING

        use_ltv = (self.graph_enc == "none")
        self.use_ltv       = use_ltv
        self.limb_obs_size = 32 if use_ltv else 26

        self.multi_variant   = (xml_paths is not None)
        self.env_variant_ids = env_variant_ids

        # ── Load graph data ───────────────────────────────────────────────
        print("[ObsBuilder] Loading graph data...", flush=True)
        if self.multi_variant:
            self.variant_data = []
            for i, xp in enumerate(xml_paths):
                up = urdf_paths[i] if urdf_paths else None
                print(f"  Loading variant {i}: "
                      f"topo={os.path.basename(xp)}  "
                      f"kin={os.path.basename(up) if up else 'same'}", flush=True)
                vd = _load_variant_graph(
                    xp, self.max_limbs, self.max_joints,
                    self.graph_enc, use_ltv, self.num_actions, device,
                    urdf_path=up)
                self.variant_data.append(vd)
                print(f"  Variant {i}: {vd.num_limbs} limbs", flush=True)

            self._ref = self.variant_data[0]
            self._stack_variant_tensors()
        else:
            vd = _load_variant_graph(
                xml_path, self.max_limbs, self.max_joints,
                self.graph_enc, use_ltv, self.num_actions, device,
                urdf_path=None)
            self.variant_data = [vd]
            self._ref = vd
            self._stack_variant_tensors()

        self._xml_path = xml_path

        # ── Foot indices from Isaac Gym ───────────────────────────────────
        bnames = env.gym.get_actor_rigid_body_names(
            env.envs[0], env.actor_handles[0])
        self.left_foot_idx  = next(
            i for i, n in enumerate(bnames) if "left_ankle_roll"  in n.lower())
        self.right_foot_idx = next(
            i for i, n in enumerate(bnames) if "right_ankle_roll" in n.lower())
        self.feet_indices = torch.tensor(
            [self.left_foot_idx, self.right_foot_idx],
            dtype=torch.long, device=device)

        # ── Spaces ────────────────────────────────────────────────────────
        prop_dim = self.max_limbs * self.limb_obs_size
        ctx_dim  = int(self._ref.context.shape[1])
        inf      = float("inf")

        obs_spaces = {
            "proprioceptive":   _BoxSpace(-inf, inf, (prop_dim,)),
            "context":          _BoxSpace(-inf, inf, (ctx_dim,)),
            "obs_padding_mask": _BoxSpace(-inf, inf, (self.max_limbs,)),
            "act_padding_mask": _BoxSpace(-inf, inf, (self.max_limbs,)),
            "edges":            _BoxSpace(-inf, inf, (self.max_joints * 2,)),
            "traversals":       _BoxSpace(-inf, inf, (self.max_limbs,)),
            "SWAT_RE":          _BoxSpace(-inf, inf,
                                         (self.max_limbs, self.max_limbs, 3)),
        }
        if self.graph_enc != "none":
            feat_dim = (7 if self.graph_enc == "onehot"
                        else 6 if self.graph_enc == "topological"
                        else cfg.MODEL.RWSE_K)   # rwse
            obs_spaces["graph_node_features"] = _BoxSpace(
                -inf, inf, (self.max_limbs, feat_dim))
            obs_spaces["graph_A_norm"] = _BoxSpace(
                0., 1., (self.max_limbs, self.max_limbs))

        self.observation_space = _DictSpace(obs_spaces)
        self.act_padding_mask  = self._ref.act_padding_mask
        self.action_space      = _BoxSpace(-1., 1., (self.max_limbs,))

        print(f"[ObsBuilder] limb_obs_size={self.limb_obs_size} "
              f"prop_dim={prop_dim} graph_enc={self.graph_enc} "
              f"variants={len(self.variant_data)}", flush=True)

    def _stack_variant_tensors(self):
        self.contexts_stacked   = torch.cat(
            [vd.context    for vd in self.variant_data], dim=0)
        self.edges_stacked      = torch.cat(
            [vd.edges      for vd in self.variant_data], dim=0)
        self.traversals_stacked = torch.cat(
            [vd.traversals for vd in self.variant_data], dim=0)
        self.obs_masks_stacked  = torch.cat(
            [vd.obs_padding_mask for vd in self.variant_data], dim=0)
        self.act_masks_stacked  = torch.cat(
            [vd.act_padding_mask for vd in self.variant_data], dim=0)
        self.joint_lo_stacked   = torch.stack(
            [vd.joint_lo   for vd in self.variant_data], dim=0)
        self.joint_hi_stacked   = torch.stack(
            [vd.joint_hi   for vd in self.variant_data], dim=0)
        self.joint_span_stacked = torch.stack(
            [vd.joint_span for vd in self.variant_data], dim=0)
        self.jrange_stacked     = torch.stack(
            [vd.jrange_norm for vd in self.variant_data], dim=0)

        if self.graph_enc != "none":
            self.gnf_stacked = torch.cat(
                [vd.graph_node_features for vd in self.variant_data], dim=0)
            self.ga_stacked  = torch.cat(
                [vd.graph_A_norm        for vd in self.variant_data], dim=0)
        else:
            self.gnf_stacked = None
            self.ga_stacked  = None

        if self.use_ltv:
            self.ltv_stacked = torch.stack(
                [vd.ltv for vd in self.variant_data], dim=0)
        else:
            self.ltv_stacked = None

        self.swat_re_stacked = torch.cat(
            [vd.swat_re for vd in self.variant_data], dim=0)

    # ── Per-step build ────────────────────────────────────────────────────────

    def build(self, commands, episode_steps, last_actions, dt):
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

        # Global state
        ang_vel_w    = root_states[:, 10:13]
        base_ang_vel = _quat_rotate_inverse(quat_wxyz, ang_vel_w) * 0.25
        grav_w       = torch.tensor(
            [0., 0., -1.], device=self.device).expand(N, 3)
        proj_gravity = _quat_rotate_inverse(quat_wxyz, grav_w)
        cmd_scaled   = commands * torch.tensor(
            [2.0, 2.0, 0.25], device=self.device)

        t           = episode_steps.float() * dt
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        sin_ph      = torch.sin(2. * math.pi * phase_left).unsqueeze(1)
        cos_ph      = torch.cos(2. * math.pi * phase_left).unsqueeze(1)
        global_state = torch.cat(
            [base_ang_vel, proj_gravity, cmd_scaled, sin_ph, cos_ph], dim=1)

        # Per-variant selection
        if self.multi_variant and self.env_variant_ids is not None:
            vid = self.env_variant_ids
        else:
            vid = torch.zeros(N, dtype=torch.long, device=self.device)

        joint_lo   = self.joint_lo_stacked[vid]
        joint_span = self.joint_span_stacked[vid]
        jrange     = self.jrange_stacked[vid]
        ctx        = self.contexts_stacked[vid]
        edges      = self.edges_stacked[vid]
        trav       = self.traversals_stacked[vid]
        swat_re    = self.swat_re_stacked[vid]
        obs_mask   = self.obs_masks_stacked[vid]
        act_mask   = self.act_masks_stacked[vid]

        # Per-body kinematics
        num_limbs = self._ref.num_limbs
        body_pos  = rigid_body[:, :num_limbs, 0:3]
        body_velp = rigid_body[:, :num_limbs, 7:10]
        body_velr = rigid_body[:, :num_limbs, 10:13]
        body_quat = _ig_quat_to_wxyz(rigid_body[:, :num_limbs, 3:7])

        root_x = body_pos[:, 0:1, 0:1]
        xpos   = body_pos.clone(); xpos[:, :, 0:1] -= root_x
        xvelp  = body_velp.clamp(-10, 10)
        xvelr  = body_velr
        expmap = _batch_quat_to_expmap(body_quat)

        angle_norm = (dof_pos - joint_lo) / joint_span

        # Assemble per-limb tokens
        tokens = torch.zeros(
            N, self.max_limbs, self.limb_obs_size, device=self.device)

        for li in range(num_limbs):
            b = 0
            tokens[:, li, b:b+3] = xpos[:, li];   b += 3
            tokens[:, li, b:b+3] = xvelp[:, li];  b += 3
            tokens[:, li, b:b+3] = xvelr[:, li];  b += 3
            tokens[:, li, b:b+3] = expmap[:, li];  b += 3

            if self.use_ltv:
                tokens[:, li, b:b+6] = self.ltv_stacked[vid, li]; b += 6

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
            "context":          ctx,
            "edges":            edges,
            "traversals":       trav,
            "SWAT_RE":          swat_re,
            "obs_padding_mask": obs_mask,
            "act_padding_mask": act_mask,
        }
        if self.graph_enc != "none":
            obs["graph_node_features"] = self.gnf_stacked[vid]
            obs["graph_A_norm"]        = self.ga_stacked[vid]

        return obs, phase_left, phase_right