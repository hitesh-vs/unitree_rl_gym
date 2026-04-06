"""
deploy/deploy_mujoco/eval_zeroshot.py

Zero-shot evaluation of a trained modular policy on new robots
(H1, G1 variants with different params, or any URDF+XML pair).

Key fixes vs record_traj_modular.py:
  1. Context is the correct 12-dim per-limb morphology vector matching training
     (_load_variant_graph in obs_builder.py). Old script used 4-dim context
     which broke FiLM conditioning at eval time.
  2. Parent array built from body_parentid directly — not inferred from joints.
     The joint-based approach fails for any robot where the root body has no
     joint pointing to it (H1: pelvis=body[1], parentid=0/world, no joint
     has bodyid=1 so parents[0] was never set to -1 → StopIteration).
  3. XML stripped correctly: mesh geoms replaced with dummy spheres so
     MuJoCo accepts the stripped file without "mesh not found" errors.
  4. TWO-MODEL SPLIT: stripped XML used only for graph topology (ZeroShotObsBuilder).
     Full URDF loaded separately as sim model for physics — this is the model
     that has actuators, so d.ctrl has the right shape.
  5. FiLM auto-detected from checkpoint keys — no --film flag needed.
  6. ob_mean/ob_var multi-variant shape (V, prop_dim) handled by averaging.
  7. Arbitrary DOF count (H1=10, G1=12) handled throughout.

Usage:
    # H1 zero-shot
    python deploy/deploy_mujoco/eval_zeroshot.py \
        --checkpoint output_film/Mar21/model_3000.pt \
        --robots '[{"name":"h1","urdf":"resources/robots/h1/h1.urdf","xml":"resources/robots/h1/h1.xml","base_height":0.98}]' \
        --graph_encoding rwse --duration 20.0 --cmd_vx 0.5

    # Held-out G1 variants
    python deploy/deploy_mujoco/eval_zeroshot.py \
        --checkpoint output_film/Mar21/model_3000.pt \
        --variants_metadata resources/robots/g1_variants_heldout/variants_metadata.json \
        --graph_encoding rwse --duration 20.0 --cmd_vx 0.5
"""

import os
import sys
import json
import math
import time
import copy
import pickle
import argparse
import tempfile
import numpy as np
import torch
import mujoco
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modular_policy.config import cfg
from modular_policy.algos.ppo.model import ActorCritic, Agent
from modular_policy.algos.ppo.obs_builder import _BoxSpace, _DictSpace
from modular_policy.graphs.parser import MujocoGraphParser

# ── Constants — must match training ──────────────────────────────────────────
GAIT_PERIOD  = 0.8
GAIT_OFFSET  = 0.5
SIM_DT       = 0.005
DECIMATION   = 4
POLICY_DT    = SIM_DT * DECIMATION   # 0.02s
ACTION_SCALE = 0.25

G1_KP = {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40}
G1_KD = {"hip_yaw": 2,   "hip_roll": 2,   "hip_pitch": 2,   "knee": 4,   "ankle": 2}

H1_KP = {"hip_yaw": 150, "hip_roll": 150, "hip_pitch": 150, "knee": 200, "ankle": 60}
H1_KD = {"hip_yaw": 3,   "hip_roll": 3,   "hip_pitch": 3,   "knee": 5,   "ankle": 3}

G1_DEFAULT_ANGLES = {
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

H1_DEFAULT_ANGLES = {
    "left_hip_yaw_joint":    0.0,
    "left_hip_roll_joint":   0.0,
    "left_hip_pitch_joint": -0.15,
    "left_knee_joint":       0.35,
    "left_ankle_joint":     -0.20,
    "right_hip_yaw_joint":   0.0,
    "right_hip_roll_joint":  0.0,
    "right_hip_pitch_joint":-0.15,
    "right_knee_joint":      0.35,
    "right_ankle_joint":    -0.20,
}

# ── Normalisation ranges — identical to obs_builder.py ───────────────────────
NORM_RANGES = {
    "mass":     (0.05,  20.0),
    "origin_x": (-0.15, 0.15),
    "origin_y": (-0.15, 0.15),
    "origin_z": (-0.35, 0.05),
    "com_z":    (-0.20, 0.20),
    "lower":    (-3.2,  0.0),
    "upper":    (0.0,   3.2),
    "effort":   (30.,   170.),
    "damping":  (0.0,   0.005),
    "axis_x":   (-1.,   1.),
    "axis_y":   (-1.,   1.),
    "axis_z":   (-1.,   1.),
}
MORPH_CTX_DIM = 12


def _norm(val, key):
    lo, hi = NORM_RANGES[key]
    return float(np.clip(2. * (val - lo) / (hi - lo + 1e-8) - 1., -1., 1.))


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_rotate_inverse_np(q, v):
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v - w * t + np.cross(np.array([x, y, z]), t)


def _quat_to_expmap(q):
    w        = np.clip(q[0], -1 + 1e-7, 1 - 1e-7)
    xyz      = q[1:]
    angle    = 2.0 * np.arccos(abs(w))
    sin_half = np.sin(angle / 2)
    if sin_half < 1e-8:
        return np.zeros(3)
    return xyz / sin_half * angle


# ── Include resolver ──────────────────────────────────────────────────────────

def _resolve_includes(root, xml_dir):
    """Recursively inline all <include> elements before stripping."""
    changed = True
    while changed:
        changed = False
        for parent in root.iter():
            for i, child in enumerate(list(parent)):
                if child.tag != "include":
                    continue
                href = child.attrib.get("file", "")
                if not os.path.isabs(href):
                    href = os.path.join(xml_dir, href)
                if not os.path.exists(href):
                    print(f"  [warn] include not found: {href} — skipping")
                    parent.remove(child)
                    changed = True
                    continue
                inc_tree = ET.parse(href)
                inc_root = inc_tree.getroot()
                parent.remove(child)
                for j, inc_child in enumerate(inc_root):
                    parent.insert(i + j, copy.deepcopy(inc_child))
                changed = True
                break


# ── XML stripping ─────────────────────────────────────────────────────────────

def strip_xml(xml_path):
    """
    Strip a full MuJoCo XML to worldbody-only topology XML.
    Used ONLY for graph topology (MujocoGraphParser, edges, traversals).
    Physics simulation always uses the original full URDF.
    """
    tree    = ET.parse(xml_path)
    root    = tree.getroot()
    xml_dir = os.path.dirname(os.path.abspath(xml_path))

    _resolve_includes(root, xml_dir)

    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ["meshdir", "assetdir"]:
            val = compiler.attrib.get(attr, "")
            if val and not os.path.isabs(val):
                compiler.set(attr, os.path.join(xml_dir, val))

    for tag in ["actuator", "sensor", "tendon", "equality",
                "contact", "keyframe", "asset"]:
        for elem in root.findall(tag):
            root.remove(elem)

    # Remove floor/ground geoms and lights from worldbody root.
    # These are direct children of <worldbody> (not inside any <body>)
    # and reference materials/textures that were just removed with <asset>.
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for g in worldbody.findall("geom"):
            worldbody.remove(g)
        for elem in worldbody.findall("light"):
            worldbody.remove(elem)

    # Replace body geoms with dummy spheres (existing loop, unchanged)
    for body in root.iter("body"):
        for g in body.findall("geom"):
            body.remove(g)
        dummy = ET.SubElement(body, "geom")
        dummy.set("type", "sphere")
        dummy.set("size", "0.01")
        dummy.set("contype", "0")
        dummy.set("conaffinity", "0")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False, mode="w", encoding="utf-8")
    tree.write(tmp.name)
    tmp.close()
    print(f"  [strip_xml] {os.path.basename(xml_path)} → {tmp.name}")
    return tmp.name


def _maybe_strip_xml(xml_path):
    if xml_path is None:
        return xml_path
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        needs_strip = any(
            root.find(tag) is not None
            for tag in ["actuator", "sensor", "tendon", "asset"]
        )
        # Also strip if worldbody has direct geom children (floor plane)
        worldbody = root.find("worldbody")
        if worldbody is not None and worldbody.findall("geom"):
            needs_strip = True
        if needs_strip:
            print(f"  Stripping {os.path.basename(xml_path)}...")
            return strip_xml(xml_path)
        return xml_path
    except Exception:
        return xml_path


# ── Morphology context — matches _load_variant_graph in obs_builder.py ────────

def build_morph_context(xml_path, urdf_path, max_limbs):
    m_topo = mujoco.MjModel.from_xml_path(xml_path)
    m_kin  = mujoco.MjModel.from_xml_path(urdf_path) if urdf_path else m_topo

    topo_names   = [m_topo.body(i).name for i in range(1, m_topo.nbody)]
    kin_names    = [m_kin.body(i).name  for i in range(1, m_kin.nbody)]
    kin_name_set = set(kin_names)
    num_limbs    = len(topo_names)

    joint_info = {}
    if urdf_path and urdf_path.endswith(".urdf"):
        urdf_tree = ET.parse(urdf_path)
        urdf_root = urdf_tree.getroot()
        for jnt in urdf_root.iter("joint"):
            jtype = jnt.attrib.get("type", "fixed")
            if jtype not in ("revolute", "prismatic"):
                continue
            child_elem = jnt.find("child")
            if child_elem is None:
                continue
            child_link = child_elem.attrib.get("link", "")
            limit   = jnt.find("limit")
            lower   = float(limit.attrib.get("lower",  "0"))  if limit is not None else 0.
            upper   = float(limit.attrib.get("upper",  "0"))  if limit is not None else 0.
            effort  = float(limit.attrib.get("effort", "88")) if limit is not None else 88.
            dyn     = jnt.find("dynamics")
            damping = float(dyn.attrib.get("damping", "0.001")) if dyn is not None else 0.001
            axis_elem = jnt.find("axis")
            ax = ([float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()]
                  if axis_elem is not None else [0., 0., 1.])
            orig_elem = jnt.find("origin")
            origin = ([float(v) for v in orig_elem.attrib.get("xyz", "0 0 0").split()]
                      if orig_elem is not None else [0., 0., 0.])
            joint_info[child_link] = {
                "lower": lower, "upper": upper, "effort": effort,
                "damping": damping, "axis": ax, "origin": origin,
            }
    else:
        for ji in range(1, m_kin.njnt):
            bname  = m_kin.body(m_kin.jnt_bodyid[ji]).name
            jrange = m_kin.jnt_range[ji]
            joint_info[bname] = {
                "lower": float(jrange[0]), "upper": float(jrange[1]),
                "effort": 88., "damping": 0.001,
                "axis": [0., 0., 1.], "origin": [0., 0., 0.],
            }

    ctx_raw = np.zeros((num_limbs, MORPH_CTX_DIM), dtype=np.float32)
    for li, bname in enumerate(topo_names[:num_limbs]):
        if bname in kin_name_set:
            kidx = kin_names.index(bname) + 1
            mass = float(m_kin.body_mass[kidx])
            com  = m_kin.body_ipos[kidx]
        else:
            tidx = topo_names.index(bname) + 1
            mass = float(m_topo.body_mass[tidx])
            com  = m_topo.body_ipos[tidx]

        ctx_raw[li, 0] = _norm(mass, "mass")
        if bname in joint_info:
            ji = joint_info[bname]
            ctx_raw[li, 1] = _norm(ji["origin"][0], "origin_x")
            ctx_raw[li, 2] = _norm(ji["origin"][1], "origin_y")
            ctx_raw[li, 3] = _norm(ji["origin"][2], "origin_z")
        ctx_raw[li, 4] = _norm(float(com[2]), "com_z")
        if bname in joint_info:
            ji = joint_info[bname]
            ctx_raw[li, 5]  = _norm(ji["lower"],   "lower")
            ctx_raw[li, 6]  = _norm(ji["upper"],   "upper")
            ctx_raw[li, 7]  = _norm(ji["effort"],  "effort")
            ctx_raw[li, 8]  = _norm(ji["damping"], "damping")
            ctx_raw[li, 9]  = _norm(ji["axis"][0], "axis_x")
            ctx_raw[li, 10] = _norm(ji["axis"][1], "axis_y")
            ctx_raw[li, 11] = _norm(ji["axis"][2], "axis_z")

    ctx_pad = np.zeros((max_limbs, MORPH_CTX_DIM), dtype=np.float32)
    ctx_pad[:num_limbs] = ctx_raw
    return ctx_pad.flatten()


# ── Parent array — built from body_parentid, not from joints ─────────────────

def build_parents(m, num_limbs):
    parents = []
    for li in range(num_limbs):
        bi     = li + 1
        par_bi = int(m.body_parentid[bi])
        if par_bi == 0:
            parents.append(-1)
        else:
            par_li = par_bi - 1
            parents.append(par_li if 0 <= par_li < num_limbs else -1)
    if not any(p == -1 for p in parents):
        print("  [warn] build_parents: no root found — forcing limb 0")
        parents[0] = -1
    return parents


def dfs_traversal(parents):
    num_limbs = len(parents)
    roots     = [i for i, p in enumerate(parents) if p == -1]
    root      = roots[0] if roots else 0
    children  = {i: [] for i in range(num_limbs)}
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


# ── ZeroShotObsBuilder ────────────────────────────────────────────────────────

class ZeroShotObsBuilder:
    """
    Builds per-limb obs dict from a MjData produced by the SIM model.

    Two separate models are used:
      topo_xml  — stripped XML, used only for graph structure (parser, edges,
                  traversals). self.m holds this model. It has no actuators.
      urdf_path — full URDF, passed as sim_model to run_robot. Has actuators.

    build(d, ...) receives MjData from the sim model, not self.m.
    self.m is only used for static graph fields in __init__.
    """

    def __init__(self, topo_xml, device, urdf_path=None, num_dof=12):
        self.device     = device
        self.graph_enc  = cfg.MODEL.GRAPH_ENCODING
        self.max_limbs  = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS
        self.num_dof    = num_dof

        use_ltv = (self.graph_enc == "none")
        self.use_ltv       = use_ltv
        self.limb_obs_size = 32 if use_ltv else 26

        # Topology model — no actuators, used only for static graph fields
        self.m = mujoco.MjModel.from_xml_path(topo_xml)

        parser         = MujocoGraphParser(topo_xml)
        self.num_limbs = parser.N
        num_pads       = self.max_limbs - self.num_limbs

        print(f"  [ZeroShotObsBuilder] topo nbody={self.m.nbody}  "
              f"num_limbs={self.num_limbs}  max_limbs={self.max_limbs}  "
              f"num_dof={num_dof}  graph_enc={self.graph_enc}")

        # ── Context: 12-dim per limb ──────────────────────────────────────
        kin_path = urdf_path if urdf_path else topo_xml
        ctx_flat = build_morph_context(topo_xml, kin_path, self.max_limbs)
        self.context = torch.tensor(
            ctx_flat, dtype=torch.float32, device=device).unsqueeze(0)

        # ── Joint ranges from kinematics model ────────────────────────────
        m_kin  = mujoco.MjModel.from_xml_path(kin_path)
        n_jnts = min(m_kin.njnt - 1, num_dof)
        jnt_range = m_kin.jnt_range[1:n_jnts+1].copy()
        if n_jnts < num_dof:
            full = np.zeros((num_dof, 2), dtype=np.float64)
            full[:n_jnts] = jnt_range
            jnt_range = full

        self.joint_lo    = jnt_range[:, 0].astype(np.float32)
        self.joint_hi    = jnt_range[:, 1].astype(np.float32)
        self.joint_span  = np.clip(self.joint_hi - self.joint_lo, 1e-8, None)
        self.jrange_norm = np.stack(
            [self.joint_lo / math.pi, self.joint_hi / math.pi], axis=1)

        # ── Edges (from topology model) ───────────────────────────────────
        joint_to   = self.m.jnt_bodyid[1:].copy() - 1
        parent_ids = self.m.body_parentid.copy()
        joint_from = np.array([parent_ids[c + 1] - 1 for c in joint_to])
        edges_raw  = np.vstack((joint_to, joint_from)).T.flatten().astype(np.int32)
        edges_pad  = np.zeros(self.max_joints * 2, dtype=np.float32)
        edges_pad[:len(edges_raw)] = edges_raw
        self.edges = torch.tensor(edges_pad, device=device).unsqueeze(0)

        # ── Traversals (from topology model) ─────────────────────────────
        parents  = build_parents(self.m, self.num_limbs)
        trav     = dfs_traversal(parents)
        trav_pad = np.zeros(self.max_limbs, dtype=np.float32)
        trav_pad[:len(trav)] = trav
        self.traversals = torch.tensor(trav_pad, device=device).unsqueeze(0)

        print(f"  [ZeroShotObsBuilder] parents={parents}")
        print(f"  [ZeroShotObsBuilder] traversal={trav.tolist()}")

        self.swat_re = torch.zeros(
            1, self.max_limbs, self.max_limbs, 3, device=device)

        obs_mask = [False]*self.num_limbs + [True]*num_pads
        act_mask = [True] + [False]*(self.num_limbs-1) + [True]*num_pads
        self.obs_padding_mask = torch.tensor(
            obs_mask, dtype=torch.bool, device=device).unsqueeze(0)
        self.act_padding_mask = torch.tensor(
            act_mask, dtype=torch.bool, device=device).unsqueeze(0)

        # ── GCN features ──────────────────────────────────────────────────
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

        # ── LTV ───────────────────────────────────────────────────────────
        if use_ltv:
            ltvs = []
            for i in range(1, self.num_limbs + 1):
                bn = mujoco.mj_id2name(
                    self.m, mujoco.mjtObj.mjOBJ_BODY, i).lower()
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

    def build(self, d, command, episode_step, last_action):
        """
        d is MjData from the SIM model (full URDF), not the topology model.
        Both models share the same body structure so xpos/xquat/cvel indices
        are identical — the topo model just lacks actuators and geom assets.
        """
        quat_wxyz    = d.qpos[3:7]
        ang_vel_w    = d.qvel[3:6]
        base_ang_vel = _quat_rotate_inverse_np(quat_wxyz, ang_vel_w) * 0.25
        proj_grav    = _quat_rotate_inverse_np(quat_wxyz, np.array([0., 0., -1.]))
        cmd_scaled   = command * np.array([2.0, 2.0, 0.25])

        t           = episode_step * POLICY_DT
        phase_left  = (t % GAIT_PERIOD) / GAIT_PERIOD
        phase_right = (phase_left + GAIT_OFFSET) % 1.0
        global_state = np.concatenate([
            base_ang_vel, proj_grav, cmd_scaled,
            [np.sin(2. * math.pi * phase_left)],
            [np.cos(2. * math.pi * phase_left)],
        ])

        dof_pos    = d.qpos[7:7 + self.num_dof]
        angle_norm = (dof_pos - self.joint_lo) / self.joint_span
        pelvis_x   = d.xpos[1, 0]

        tokens = np.zeros(
            (self.max_limbs, self.limb_obs_size), dtype=np.float32)

        for li in range(self.num_limbs):
            bi   = li + 1
            xpos = d.xpos[bi].copy(); xpos[0] -= pelvis_x
            # Rotate world-frame cvel into body frame.
            # Training used Isaac Gym rigid_body_states[:, :, 7:13] which are
            # body-frame velocities. MuJoCo cvel is world-frame spatial velocity
            # (indices 0:3 = rotational, 3:6 = translational). Rotating to body
            # frame with xmat fixes the millions-scale values seen at test time.
            R          = d.xmat[bi].reshape(3, 3)          # world→body rotation
            xvelp      = np.clip(R.T @ d.cvel[bi, 3:6], -10, 10)  # linear, body frame
            xvelr      = R.T @ d.cvel[bi, 0:3]                      # angular, body frame

            expmap = _quat_to_expmap(d.xquat[bi].copy())

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


# ── PD helpers ────────────────────────────────────────────────────────────────

def get_kp_kd_arrays(m, num_dof, kp_map, kd_map):
    kp_arr = np.zeros(num_dof, dtype=np.float32)
    kd_arr = np.zeros(num_dof, dtype=np.float32)
    for i in range(num_dof):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        if jname is None:
            continue
        jname = jname.lower()
        for key in kp_map:
            if key in jname:
                kp_arr[i] = kp_map[key]
                kd_arr[i] = kd_map[key]
                break
    return kp_arr, kd_arr


def get_default_angles(m, num_dof, default_map):
    default = np.zeros(num_dof, dtype=np.float32)
    for i in range(num_dof):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)
        if jname and jname in default_map:
            default[i] = default_map[jname]
    return default


def count_actuated_dof(m):
    return m.njnt - 1


def detect_robot_type(path):
    return "h1" if "h1" in (path or "").lower() else "g1"


def get_robot_pd_defaults(robot_type):
    if robot_type == "h1":
        return H1_KP, H1_KD, H1_DEFAULT_ANGLES
    return G1_KP, G1_KD, G1_DEFAULT_ANGLES


# ── Obs normalisation warmup ──────────────────────────────────────────────────

def compute_obs_stats_warmup(obs_builder, m_sim, default_angles,
                              kp_arr, kd_arr, base_height, command,
                              num_dof, warmup_seconds=5.0):
    """
    Random-action warmup using m_sim (full URDF with actuators).
    obs_builder.build() receives MjData from m_sim — body indices match
    because both models share the same kinematic structure.
    """
    print(f"  Warmup ({warmup_seconds}s)...")
    d = mujoco.MjData(m_sim)
    mujoco.mj_resetData(m_sim, d)
    d.qpos[2]  = base_height + 0.02
    d.qpos[3]  = 1.0
    d.qpos[7:7+num_dof] = default_angles
    mujoco.mj_forward(m_sim, d)

    prop_dim = cfg.MODEL.MAX_LIMBS * obs_builder.limb_obs_size
    ob_mean  = torch.zeros(prop_dim)
    ob_var   = torch.ones(prop_dim)
    ob_count = torch.tensor(1e-4)

    target_dof_pos = default_angles.copy()
    last_action    = np.zeros(num_dof, dtype=np.float32)
    episode_step   = 0

    for sim_step in range(int(warmup_seconds / SIM_DT)):
        tau = ((target_dof_pos - d.qpos[7:7+num_dof]) * kp_arr +
               (np.zeros(num_dof) - d.qvel[6:6+num_dof]) * kd_arr)
        # m_sim has actuators so d.ctrl has correct shape
        d.ctrl[:num_dof] = tau
        mujoco.mj_step(m_sim, d)

        if sim_step % DECIMATION != 0:
            continue

        obs, _, _ = obs_builder.build(d, command, episode_step, last_action)
        prop = obs["proprioceptive"][0].float()

        d_   = prop - ob_mean
        tot  = ob_count + 1
        ob_mean  = ob_mean + d_ / tot
        ob_var   = (ob_var * ob_count + d_.pow(2) * ob_count / tot) / tot
        ob_count = tot

        if sim_step % (DECIMATION * 10) == 0:
            target_dof_pos = (default_angles +
                np.random.uniform(-0.3, 0.3, num_dof).astype(np.float32))

        if d.qpos[2] < base_height * 0.4 or d.qpos[2] > base_height * 1.8:
            mujoco.mj_resetData(m_sim, d)
            d.qpos[2]  = base_height + 0.02
            d.qpos[3]  = 1.0
            d.qpos[7:7+num_dof] = default_angles
            mujoco.mj_forward(m_sim, d)
            target_dof_pos = default_angles.copy()
            episode_step   = 0
        else:
            episode_step += 1

    print(f"  Warmup done: mean_abs={ob_mean.abs().mean():.4f}  "
          f"var_mean={ob_var.mean():.4f}")
    return ob_mean, ob_var


# ── Single robot eval ─────────────────────────────────────────────────────────

def run_robot(robot_spec, actor_critic, agent, command, duration, device):
    bht     = robot_spec["base_height_target"]
    num_dof = robot_spec["num_dof"]
    clipob  = 10.0

    # Topology model: stripped XML, no actuators, for graph fields only
    obs_builder = ZeroShotObsBuilder(
        robot_spec["xml_path"], device,
        urdf_path=robot_spec.get("urdf_path"),
        num_dof=num_dof)

    # Sim model: base stripped XML which has floating base + actuators
    # Variant URDF cannot be used — it has no free joint
    sim_path = robot_spec["sim_xml_path"]
    m_sim    = mujoco.MjModel.from_xml_path(sim_path)
    m_sim.opt.timestep = SIM_DT
    d = mujoco.MjData(m_sim)

    kp_arr, kd_arr = get_kp_kd_arrays(
        m_sim, num_dof, robot_spec["kp_map"], robot_spec["kd_map"])
    default_angles = get_default_angles(
        m_sim, num_dof, robot_spec["default_angles_map"])

    print(f"  sim={os.path.basename(sim_path)}  nq={m_sim.nq}  nu={m_sim.nu}")
    # rest of function unchanged ...

    # Warmup uses m_sim so d.ctrl has the right shape
    ob_mean, ob_var = compute_obs_stats_warmup(
        obs_builder, m_sim, default_angles, kp_arr, kd_arr,
        bht, command, num_dof, warmup_seconds=5.0)
    ob_mean = ob_mean.to(device)
    ob_var  = ob_var.to(device)

    def normalize_obs(obs_dict):
        prop = obs_dict["proprioceptive"]
        obs_dict["proprioceptive"] = (
            (prop - ob_mean) / (ob_var + 1e-8).sqrt()
        ).clamp(-clipob, clipob)
        return obs_dict

    mujoco.mj_resetData(m_sim, d)
    d.qpos[2]  = bht + 0.02
    d.qpos[3]  = 1.0
    d.qpos[7:7+num_dof] = default_angles
    mujoco.mj_forward(m_sim, d)

    h_lo = bht * 0.5
    h_hi = bht * 1.6

    max_sim_steps   = int(duration / SIM_DT)
    trajectory      = []
    last_action     = np.zeros(num_dof, dtype=np.float32)
    target_dof_pos  = default_angles.copy()
    episode_step    = 0
    sim_step        = 0
    total_reward    = 0.0
    ep_len          = 0
    episodes        = 0
    episode_rewards = []

    while sim_step < max_sim_steps:
        tau = ((target_dof_pos - d.qpos[7:7+num_dof]) * kp_arr +
               (np.zeros(num_dof) - d.qvel[6:6+num_dof]) * kd_arr)
        d.ctrl[:num_dof] = tau   # works because m_sim has actuators
        mujoco.mj_step(m_sim, d)
        sim_step += 1

        # Record every sim step for playback
        trajectory.append({
            "qpos": d.qpos.copy(),
            "qvel": d.qvel.copy(),
            "time": d.time,
        })

        if sim_step % DECIMATION != 0:
            continue

        # obs_builder.build receives d from m_sim — body indices are the same
        obs, _, _ = obs_builder.build(d, command, episode_step, last_action)
        obs = normalize_obs(obs)

        with torch.no_grad():
            _, act, _, _, _ = agent.act(obs, unimal_ids=[0])

        act_mask       = obs_builder.act_padding_mask[0].bool()
        real_action    = act[0, ~act_mask].cpu().numpy()[:num_dof]
        target_dof_pos = real_action * ACTION_SCALE + default_angles
        last_action    = real_action.copy()
        episode_step  += 1

        r = 0.15 * POLICY_DT
        base_vel_body = _quat_rotate_inverse_np(d.qpos[3:7], d.qvel[:3])
        lin_err = np.sum((command[:2] - base_vel_body[:2])**2)
        r += 1.0 * np.exp(-lin_err / 0.25) * POLICY_DT
        total_reward += r
        ep_len       += 1

        height = d.qpos[2]
        w, x, y, z = d.qpos[3:7]
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        fell  = (height < h_lo or height > h_hi or
                 abs(pitch) > 1.0 or abs(roll) > 0.8)

        if fell:
            episode_rewards.append({
                "episode":      episodes + 1,
                "return":       round(total_reward, 3),
                "length_secs":  round(ep_len * POLICY_DT, 2),
                "length_steps": ep_len,
                "termination":  "fall",
            })
            mujoco.mj_resetData(m_sim, d)
            d.qpos[2]  = bht + 0.02
            d.qpos[3]  = 1.0
            d.qpos[7:7+num_dof] = default_angles
            mujoco.mj_forward(m_sim, d)
            last_action    = np.zeros(num_dof, dtype=np.float32)
            target_dof_pos = default_angles.copy()
            episode_step   = 0
            total_reward   = 0.0
            ep_len         = 0
            episodes      += 1

    if ep_len > 0:
        episode_rewards.append({
            "episode":      episodes + 1,
            "return":       round(total_reward, 3),
            "length_secs":  round(ep_len * POLICY_DT, 2),
            "length_steps": ep_len,
            "termination":  "timeout",
        })

    return episode_rewards, trajectory


# ── Robot spec builders ───────────────────────────────────────────────────────

def build_robot_specs_from_metadata(meta_path, base_xml_override=None):
    with open(meta_path) as f:
        meta = json.load(f)
    specs     = []
    tmp_files = []
    for name, m in meta.items():
        urdf_path    = m.get("urdf")
        xml_path     = base_xml_override or m.get("xml", urdf_path)
        bht          = m.get("base_height_target", 0.78)

        # topo_xml: stripped XML for graph topology (MujocoGraphParser)
        topo_xml  = _maybe_strip_xml(xml_path)
        if topo_xml != xml_path:
            tmp_files.append(topo_xml)

        # sim_xml: the stripped XML WITH floating base + actuators for physics
        # This is always the base stripped XML — variant URDF has no free joint
        sim_xml = xml_path   # already stripped, has floating base + actuators

        robot_type = detect_robot_type(urdf_path or xml_path)
        kp_map, kd_map, def_map = get_robot_pd_defaults(robot_type)

        # DOF count from sim model (the one that has the free joint)
        m_sim   = mujoco.MjModel.from_xml_path(sim_xml)
        num_dof = m_sim.njnt - 1   # exclude free joint

        specs.append({
            "name":               name,
            "xml_path":           topo_xml,   # stripped, no actuators — topology only
            "sim_xml_path":       sim_xml,    # stripped WITH actuators — for physics
            "urdf_path":          urdf_path,  # for morphology context (mass etc)
            "base_height_target": bht,
            "num_dof":            num_dof,
            "kp_map":             kp_map,
            "kd_map":             kd_map,
            "default_angles_map": def_map,
        })
        print(f"  Spec: {name}  dof={num_dof}  bht={bht}")
    return specs, tmp_files


def build_robot_specs_from_json(robots_json):
    robots    = json.loads(robots_json)
    specs     = []
    tmp_files = []
    for r in robots:
        urdf_path = r.get("urdf")
        xml_path  = r.get("xml", urdf_path)
        bht       = r.get("base_height", 0.78)
        stripped  = _maybe_strip_xml(xml_path)
        if stripped != xml_path:
            tmp_files.append(stripped)
        robot_type = detect_robot_type(urdf_path or xml_path)
        kp_map  = r.get("kp_map")  or (H1_KP  if robot_type == "h1" else G1_KP)
        kd_map  = r.get("kd_map")  or (H1_KD  if robot_type == "h1" else G1_KD)
        def_map = (r.get("default_angles") or
                   (H1_DEFAULT_ANGLES if robot_type == "h1" else G1_DEFAULT_ANGLES))
        # Count DOF from the FULL model (URDF), not the stripped topo XML
        full_path = urdf_path if urdf_path else xml_path
        m_full    = mujoco.MjModel.from_xml_path(full_path)
        num_dof   = count_actuated_dof(m_full)
        specs.append({
            "name": r["name"], "xml_path": stripped, "urdf_path": urdf_path,
            "base_height_target": bht, "num_dof": num_dof,
            "kp_map": kp_map, "kd_map": kd_map, "default_angles_map": def_map,
        })
        print(f"  Spec: {r['name']}  dof={num_dof}  bht={bht}  type={robot_type}")
    return specs, tmp_files


# ── Policy loader ─────────────────────────────────────────────────────────────

def load_policy(checkpoint, graph_encoding, device, prop_dim, ctx_dim):
    ckpt = torch.load(checkpoint, map_location=device)

    has_film = any("film_generator" in k
                   for k in ckpt["model_state_dict"].keys())
    cfg.MODEL.TRANSFORMER.USE_FILM = has_film
    print(f"  FiLM in checkpoint: {has_film}")

    inf = float("inf")
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
    if graph_encoding != "none":
        fd = 7 if graph_encoding == "onehot" else 6
        obs_spaces["graph_node_features"] = _BoxSpace(
            -inf, inf, (cfg.MODEL.MAX_LIMBS, fd))
        obs_spaces["graph_A_norm"] = _BoxSpace(
            0., 1., (cfg.MODEL.MAX_LIMBS, cfg.MODEL.MAX_LIMBS))

    observation_space = _DictSpace(obs_spaces)
    action_space      = _BoxSpace(-1., 1., (cfg.MODEL.MAX_LIMBS,))

    actor_critic = ActorCritic(observation_space, action_space).to(device)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()
    agent = Agent(actor_critic)

    ob_mean = ckpt.get("ob_mean", torch.zeros(prop_dim))
    ob_var  = ckpt.get("ob_var",  torch.ones(prop_dim))
    if ob_mean.dim() == 2:
        ob_mean = ob_mean.mean(0)
        ob_var  = ob_var.mean(0)
    ob_mean = ob_mean.to(device)
    ob_var  = ob_var.to(device)

    return actor_critic, agent, ob_mean, ob_var


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variants_metadata", default=None)
    parser.add_argument("--robots", default=None,
                        help='JSON: [{"name":"h1","urdf":"h1.urdf",'
                             '"xml":"h1.xml","base_height":0.98}]')
    parser.add_argument("--graph_encoding", default="rwse",
                        choices=["none", "onehot", "topological", "rwse"])
    parser.add_argument("--duration",  type=float, default=20.0)
    parser.add_argument("--cmd_vx",   type=float, default=0.5)
    parser.add_argument("--cmd_vy",   type=float, default=0.0)
    parser.add_argument("--cmd_yaw",  type=float, default=0.0)
    parser.add_argument("--out_dir",  default="eval_results")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--base_xml", default=None,
                        help="Override xml path in metadata for all G1 variants "
                             "(use when metadata has stale/wrong xml paths)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device  = torch.device(args.device)
    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)

    cfg.MODEL.GRAPH_ENCODING = args.graph_encoding
    cfg.MODEL.MAX_LIMBS      = 13
    cfg.MODEL.MAX_JOINTS     = 12
    cfg.PPO.NUM_ENVS         = 1
    cfg.PPO.BATCH_SIZE       = 1
    cfg.ENV.WALKERS          = ["g1"]

    limb_obs_size = 26 if args.graph_encoding != "none" else 32
    prop_dim      = cfg.MODEL.MAX_LIMBS * limb_obs_size
    ctx_dim       = cfg.MODEL.MAX_LIMBS * MORPH_CTX_DIM  # 13 * 12 = 156

    print(f"\nLoading: {args.checkpoint}")
    actor_critic, agent, ob_mean, ob_var = load_policy(
        args.checkpoint, args.graph_encoding, device, prop_dim, ctx_dim)

    tmp_files = []
    if args.variants_metadata:
        print(f"\nRobots from: {args.variants_metadata}")
        robot_specs, tmp_files = build_robot_specs_from_metadata(
            args.variants_metadata, base_xml_override=args.base_xml)
    elif args.robots:
        print(f"\nRobots from --robots JSON")
        robot_specs, tmp_files = build_robot_specs_from_json(args.robots)
    else:
        parser.error("Provide --variants_metadata or --robots")

    summary = {}
    for spec in robot_specs:
        name = spec["name"]
        print(f"\n{'='*60}")
        print(f"Robot: {name}  dof={spec['num_dof']}  "
              f"bht={spec['base_height_target']}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            episode_rewards, trajectory = run_robot(
                spec, actor_critic, agent, command, args.duration, device)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            summary[name] = {"error": str(e)}
            continue

        elapsed = time.time() - t0

        if episode_rewards:
            returns = [e["return"]      for e in episode_rewards]
            lengths = [e["length_secs"] for e in episode_rewards]
            falls   = sum(1 for e in episode_rewards
                          if e["termination"] == "fall")
            summary[name] = {
                "episodes":      len(episode_rewards),
                "mean_return":   round(float(np.mean(returns)), 3),
                "mean_ep_len_s": round(float(np.mean(lengths)), 2),
                "max_ep_len_s":  round(float(np.max(lengths)),  2),
                "fall_rate":     round(falls / len(episode_rewards), 3),
                "elapsed_s":     round(elapsed, 1),
            }
            print(f"  episodes={len(episode_rewards)}  "
                  f"mean_return={summary[name]['mean_return']:.2f}  "
                  f"mean_len={summary[name]['mean_ep_len_s']:.1f}s  "
                  f"fall_rate={summary[name]['fall_rate']:.2f}")
        else:
            summary[name] = {"episodes": 0}

        out_pkl = os.path.join(args.out_dir, f"eval_{name}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump({"episode_rewards": episode_rewards, "trajectory": trajectory,"spec": spec}, f)

    print(f"\n{'='*72}")
    print(f"{'Robot':<24} {'Eps':>4} {'MeanRet':>8} "
          f"{'MeanLen(s)':>11} {'MaxLen(s)':>10} {'FallRate':>9}")
    print(f"{'-'*72}")
    for name, s in summary.items():
        if "error" in s:
            print(f"{name:<24}  ERROR: {s['error']}")
        elif s.get("episodes", 0) == 0:
            print(f"{name:<24}    0  (no data)")
        else:
            print(f"{name:<24} {s['episodes']:>4} {s['mean_return']:>8.2f} "
                  f"{s['mean_ep_len_s']:>11.1f} {s['max_ep_len_s']:>10.1f} "
                  f"{s['fall_rate']:>9.3f}")
    print(f"{'='*72}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")

    for p in tmp_files:
        try:
            os.unlink(p)
        except Exception:
            pass


if __name__ == "__main__":
    main()