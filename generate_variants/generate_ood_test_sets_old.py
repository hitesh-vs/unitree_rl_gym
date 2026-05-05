"""
generate_ood_test_sets.py

Generates 6 sets of OOD test variants for zero-shot evaluation.
Each set is in its OWN subfolder with its own variants_metadata.json.
This means each eval run only sees 5+1 variants — no ob_mean index errors.

Structure:
    g1_ood_test_sets/
        damping_only/
            variants_metadata.json   (6 entries: 5 robots + base)
            damping_only_robot_0.urdf
            damping_only_robot_0_changes.json
            ...
        mass_only/
            variants_metadata.json
            ...
        length_only/  ...
        armature_only/ ...
        joint_range_only/ ...
        all_combined/ ...

Sets:
  1. damping_only     — only damping varies, rest = base G1
  2. mass_only        — only mass varies (with L/R asymmetry), rest = base
  3. length_only      — only leg length varies, rest = base
  4. armature_only    — only armature varies, rest = base
  5. joint_range_only — only joint range varies, rest = base
  6. all_combined     — all params vary, close to variant_4 profile

Usage:
    python generate_ood_test_sets.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_ood_test_sets \
        --base_xml  /path/to/g1_12dof_stripped.xml \
        --seed 7777

Eval one set:
    python record_traj_isaac.py \
        --checkpoint output_film_wide/.../model_400.pt \
        --xml_path /path/to/g1_12dof_stripped.xml \
        --variants_metadata resources/robots/g1_ood_test_sets/mass_only/variants_metadata.json \
        --variant_name mass_only_robot_0 \
        --num_eval_rollouts 5 \
        --out traj_film_mass_only_0.pkl
"""

import os
import json
import copy
import argparse
import numpy as np
import xml.etree.ElementTree as ET

# ── Joint groups ──────────────────────────────────────────────────────────────
LEFT_LEG_LINKS  = {
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
}
RIGHT_LEG_LINKS = {
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
}
TORSO_LINKS     = {"pelvis"}
ACTUATED_JOINTS = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
}

BASE_DAMPING  = 0.001
BASE_ARMATURE = 0.01
BASE_HEIGHT   = 0.78


def indent(elem, level=0):
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def make_per_joint_config(rng, left_range=1.0, right_range=1.0,
                           left_effort=1.0, right_effort=1.0,
                           left_damp=1.0, right_damp=1.0,
                           left_arm=1.0, right_arm=1.0,
                           noise=0.05):
    """Build per_joint config with small noise around given scales."""
    pj = {}
    for jname in ACTUATED_JOINTS:
        is_left = jname.startswith("left")
        def n(v):
            return float(np.clip(v * (1 + rng.uniform(-noise, noise)), 0.3, 2.5))
        pj[jname] = {
            "range_scale":  n(left_range  if is_left else right_range),
            "effort_scale": n(left_effort if is_left else right_effort),
            "damp_scale":   n(left_damp   if is_left else right_damp),
            "arm_scale":    n(left_arm    if is_left else right_arm),
        }
    return pj


def apply_variant(base_root, config):
    """Apply property configuration to base URDF."""
    root = copy.deepcopy(base_root)

    left_mass  = config.get("left_mass_scale",  1.0)
    right_mass = config.get("right_mass_scale", 1.0)
    torso_mass = config.get("torso_mass_scale", 1.0)
    per_joint  = config.get("per_joint", {})

    # Mass scaling
    for link in root.iter("link"):
        lname    = link.attrib.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            continue
        if lname in LEFT_LEG_LINKS:
            scale = left_mass
        elif lname in RIGHT_LEG_LINKS:
            scale = right_mass
        elif lname in TORSO_LINKS:
            scale = torso_mass
        else:
            scale = 1.0
        orig = float(mass_elem.attrib.get("value", "1.0"))
        mass_elem.set("value", f"{orig * scale:.6g}")
        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            for attr in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                if attr in inertia_elem.attrib:
                    inertia_elem.set(attr,
                        f"{float(inertia_elem.attrib[attr]) * scale:.10g}")

    # Joint parameter scaling
    child_to_arm = {}
    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue
        jp = per_joint.get(jname, {})

        # Range + effort
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            rs   = jp.get("range_scale", 1.0)
            lo   = float(limit_elem.attrib.get("lower", "0"))
            hi   = float(limit_elem.attrib.get("upper", "0"))
            ctr  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * rs
            limit_elem.set("lower", f"{ctr - half:.8g}")
            limit_elem.set("upper", f"{ctr + half:.8g}")
            es   = jp.get("effort_scale", 1.0)
            orig = float(limit_elem.attrib.get("effort", "88"))
            limit_elem.set("effort", f"{orig * es:.4g}")

        # Damping
        damp_val = BASE_DAMPING * jp.get("damp_scale", 1.0)
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping",  f"{damp_val:.6g}")
        dyn.set("friction", "0.0")

        # Armature — store for link lookup
        child_elem = joint.find("child")
        if child_elem is not None:
            child_to_arm[child_elem.attrib.get("link", "")] = \
                BASE_ARMATURE * jp.get("arm_scale", 1.0)

    # Apply armature to link inertias
    for link in root.iter("link"):
        lname = link.attrib.get("name", "")
        if lname not in child_to_arm:
            continue
        arm_val  = child_to_arm[lname]
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia_elem = inertial.find("inertia")
        if inertia_elem is None:
            continue
        for attr in ["ixx", "iyy", "izz"]:
            orig = float(inertia_elem.attrib.get(attr, "0"))
            inertia_elem.set(attr, f"{orig + arm_val * 1e-4:.10g}")

    return root


def compute_leg_z(variant_root):
    left_leg_z = right_leg_z = 0.0
    for joint in variant_root.iter("joint"):
        jname  = joint.attrib.get("name", "")
        origin = joint.find("origin")
        if origin is None:
            continue
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        if jname in {"left_hip_pitch_joint", "left_knee_joint",
                     "left_ankle_pitch_joint", "left_ankle_roll_joint"}:
            left_leg_z  += abs(xyz[2])
        elif jname in {"right_hip_pitch_joint", "right_knee_joint",
                       "right_ankle_pitch_joint", "right_ankle_roll_joint"}:
            right_leg_z += abs(xyz[2])
    return left_leg_z, right_leg_z


def generate_set(set_name, configs, base_root, base_urdf_path,
                 base_xml, base_leg_z, out_base_dir):
    """
    Generate one set of variants in its own subfolder.
    configs: list of (config_dict, changes_dict) tuples, one per robot.
    Returns path to the set's variants_metadata.json.
    """
    set_dir = os.path.join(out_base_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)

    # Copy base URDF into this set's folder
    import shutil
    base_out = os.path.join(set_dir, "g1_12dof.urdf")
    shutil.copy(base_urdf_path, base_out)

    meta = {}

    # Add base robot first
    meta["g1_12dof"] = {
        "urdf":                  base_out,
        "xml":                   base_xml,
        "leg_z_left":            round(base_leg_z, 6),
        "leg_z_right":           round(base_leg_z, 6),
        "leg_scale":             1.0,
        "base_height_target":    BASE_HEIGHT,
        "feet_clearance_target": 0.08,
        "set":                   "base",
        "asymmetric":            False,
    }

    for i, (cfg, changes) in enumerate(configs):
        vname    = f"{set_name}_robot_{i}"
        out_urdf = os.path.join(set_dir, f"{vname}.urdf")
        out_json = os.path.join(set_dir, f"{vname}_changes.json")

        vroot = apply_variant(base_root, cfg)
        indent(vroot)
        ET.ElementTree(vroot).write(
            out_urdf, xml_declaration=True, encoding="unicode")
        with open(out_json, "w") as f:
            json.dump(changes, f, indent=2)

        left_leg_z, right_leg_z = compute_leg_z(vroot)
        avg_leg_z = min(left_leg_z, right_leg_z)
        leg_scale = avg_leg_z / base_leg_z if base_leg_z > 0 else 1.0
        bht       = round(BASE_HEIGHT * leg_scale, 4)
        fct       = round(0.08 * leg_scale, 4)

        meta[vname] = {
            "urdf":                  out_urdf,
            "xml":                   base_xml,
            "leg_z_left":            round(left_leg_z, 6),
            "leg_z_right":           round(right_leg_z, 6),
            "leg_scale":             round(leg_scale, 6),
            "base_height_target":    bht,
            "feet_clearance_target": fct,
            "set":                   set_name,
            "asymmetric":            True,
        }

    meta_path = os.path.join(set_dir, "variants_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  → {set_dir}/  ({len(configs)} robots + base)")
    return meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_urdf",   required=True)
    parser.add_argument("--out_dir",     default="resources/robots/g1_ood_test_sets")
    parser.add_argument("--base_xml",    default="")
    parser.add_argument("--seed",        type=int, default=7777)
    parser.add_argument("--num_per_set", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    N   = args.num_per_set

    tree      = ET.parse(args.base_urdf)
    base_root = tree.getroot()

    # Compute base leg Z
    base_leg_z = 0.0
    for joint in base_root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname in {"left_hip_pitch_joint", "left_knee_joint",
                     "left_ankle_pitch_joint", "left_ankle_roll_joint"}:
            origin = joint.find("origin")
            if origin is not None:
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                base_leg_z += abs(xyz[2])
    if base_leg_z < 1e-6:
        base_leg_z = 0.1027 + 0.17734 + 0.30001 + 0.017558

    print(f"\n{'='*65}")
    print(f"Generating 6 OOD test sets × {N} robots each")
    print(f"Each set in its own subfolder with its own variants_metadata.json")
    print(f"Out dir: {args.out_dir}")
    print(f"{'='*65}\n")

    all_meta_paths = {}

    # ── SET 1: damping_only ───────────────────────────────────────────────
    print("SET 1: damping_only")
    configs = []
    for i in range(N):
        left_damp  = float(rng.uniform(1.2, 1.8))
        right_damp = float(rng.uniform(0.6, 1.2))
        pj = make_per_joint_config(
            rng, left_damp=left_damp, right_damp=right_damp, noise=0.08)
        cfg     = {"per_joint": pj}
        changes = {
            "set": "damping_only", "varied_prop": ["damping"],
            "left_damp_scale": left_damp, "right_damp_scale": right_damp,
            "per_joint": {j: {"damp_scale": pj[j]["damp_scale"]} for j in pj},
        }
        configs.append((cfg, changes))
        print(f"  robot_{i}: left_damp={left_damp:.2f}x  right_damp={right_damp:.2f}x")
    all_meta_paths["damping_only"] = generate_set(
        "damping_only", configs, base_root, args.base_urdf,
        args.base_xml, base_leg_z, args.out_dir)

    # ── SET 2: mass_only ──────────────────────────────────────────────────
    print("\nSET 2: mass_only")
    configs = []
    for i in range(N):
        left_mass  = float(rng.uniform(1.8, 2.5))
        right_mass = float(rng.uniform(1.2, 1.8))
        torso_mass = float(rng.uniform(1.5, 2.2))
        pj  = make_per_joint_config(rng, noise=0.0)
        cfg = {
            "left_mass_scale":  left_mass,
            "right_mass_scale": right_mass,
            "torso_mass_scale": torso_mass,
            "per_joint": pj,
        }
        changes = {
            "set": "mass_only", "varied_prop": ["mass"],
            "group_mass_scale": {
                "left_leg": left_mass, "right_leg": right_mass, "torso": torso_mass},
        }
        configs.append((cfg, changes))
        print(f"  robot_{i}: left={left_mass:.2f}x  right={right_mass:.2f}x  "
              f"ratio={left_mass/right_mass:.2f}x")
    all_meta_paths["mass_only"] = generate_set(
        "mass_only", configs, base_root, args.base_urdf,
        args.base_xml, base_leg_z, args.out_dir)

    # ── SET 3: length_only ────────────────────────────────────────────────
    print("\nSET 3: length_only")
    configs = []
    for i in range(N):
        shared = float(rng.uniform(0.80, 1.20))
        ll     = float(np.clip(shared * rng.uniform(0.95, 1.05), 0.78, 1.22))
        rl     = float(np.clip(shared * rng.uniform(0.95, 1.05), 0.78, 1.22))
        pj     = make_per_joint_config(rng, noise=0.0)
        cfg    = {"per_joint": pj}
        changes = {
            "set": "length_only", "varied_prop": ["length"],
            "group_length_scale": {"left_leg": ll, "right_leg": rl},
        }
        configs.append((cfg, changes))
        print(f"  robot_{i}: left_len={ll:.3f}x  right_len={rl:.3f}x")
    all_meta_paths["length_only"] = generate_set(
        "length_only", configs, base_root, args.base_urdf,
        args.base_xml, base_leg_z, args.out_dir)

    # ── SET 4: armature_only ──────────────────────────────────────────────
    print("\nSET 4: armature_only")
    configs = []
    for i in range(N):
        left_arm  = float(rng.uniform(1.2, 1.8))
        right_arm = float(rng.uniform(0.7, 1.3))
        pj = make_per_joint_config(
            rng, left_arm=left_arm, right_arm=right_arm, noise=0.08)
        cfg     = {"per_joint": pj}
        changes = {
            "set": "armature_only", "varied_prop": ["armature"],
            "left_arm_scale": left_arm, "right_arm_scale": right_arm,
            "per_joint": {j: {"arm_scale": pj[j]["arm_scale"]} for j in pj},
        }
        configs.append((cfg, changes))
        print(f"  robot_{i}: left_arm={left_arm:.2f}x  right_arm={right_arm:.2f}x")
    all_meta_paths["armature_only"] = generate_set(
        "armature_only", configs, base_root, args.base_urdf,
        args.base_xml, base_leg_z, args.out_dir)

    # ── SET 5: joint_range_only ───────────────────────────────────────────
    print("\nSET 5: joint_range_only")
    configs = []
    for i in range(N):
        left_range  = float(rng.uniform(0.65, 0.90))
        right_range = float(rng.uniform(0.85, 1.30))
        pj = make_per_joint_config(
            rng, left_range=left_range, right_range=right_range, noise=0.08)
        cfg     = {"per_joint": pj}
        changes = {
            "set": "joint_range_only", "varied_prop": ["joint_range"],
            "left_range_scale": left_range, "right_range_scale": right_range,
            "per_joint": {j: {"range_scale": pj[j]["range_scale"]} for j in pj},
        }
        configs.append((cfg, changes))
        print(f"  robot_{i}: left_range={left_range:.2f}x  right_range={right_range:.2f}x")
    all_meta_paths["joint_range_only"] = generate_set(
        "joint_range_only", configs, base_root, args.base_urdf,
        args.base_xml, base_leg_z, args.out_dir)

    # ── SET 6: all_combined — close to variant_4 ─────────────────────────
    print("\nSET 6: all_combined (close to variant_4 profile)")
    configs = []
    for i in range(N):
        left_mass    = float(rng.uniform(2.0, 2.5))
        right_mass   = float(rng.uniform(1.3, 1.8))
        torso_mass   = float(rng.uniform(1.6, 2.2))
        left_effort  = float(rng.uniform(0.50, 0.70))
        right_effort = float(rng.uniform(0.90, 1.30))
        left_range   = float(rng.uniform(0.65, 0.85))
        right_range  = float(rng.uniform(0.90, 1.35))
        left_damp    = float(rng.uniform(1.2, 1.8))
        right_damp   = float(rng.uniform(0.6, 1.2))
        left_arm     = float(rng.uniform(1.1, 1.7))
        right_arm    = float(rng.uniform(0.7, 1.3))
        pj = make_per_joint_config(
            rng,
            left_range=left_range,   right_range=right_range,
            left_effort=left_effort, right_effort=right_effort,
            left_damp=left_damp,     right_damp=right_damp,
            left_arm=left_arm,       right_arm=right_arm,
            noise=0.06,
        )
        cfg = {
            "left_mass_scale":  left_mass,
            "right_mass_scale": right_mass,
            "torso_mass_scale": torso_mass,
            "per_joint": pj,
        }
        changes = {
            "set": "all_combined",
            "varied_prop": ["mass", "joint_range", "effort", "damping", "armature"],
            "note": "close to variant_4 — heavy+weak+restricted left side",
            "group_mass_scale": {
                "left_leg": left_mass, "right_leg": right_mass, "torso": torso_mass},
            "left_effort_avg":  left_effort,
            "right_effort_avg": right_effort,
            "left_range_avg":   left_range,
            "right_range_avg":  right_range,
            "per_joint": {j: pj[j] for j in pj},
        }
        configs.append((cfg, changes))
        print(f"  robot_{i}: left_mass={left_mass:.2f}x  left_effort={left_effort:.2f}x  "
              f"left_range={left_range:.2f}x")
    all_meta_paths["all_combined"] = generate_set(
        "all_combined", configs, base_root, args.base_urdf,
        args.base_xml, base_leg_z, args.out_dir)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Done. Folder structure:")
    for set_name, meta_path in all_meta_paths.items():
        print(f"  {args.out_dir}/{set_name}/variants_metadata.json")
    print(f"\nEval example:")
    print(f"  python record_traj_isaac.py \\")
    print(f"    --checkpoint output_film_wide/.../model_400.pt \\")
    print(f"    --xml_path /path/to/g1_12dof_stripped.xml \\")
    print(f"    --variants_metadata {args.out_dir}/mass_only/variants_metadata.json \\")
    print(f"    --variant_name mass_only_robot_0 \\")
    print(f"    --num_eval_rollouts 5 \\")
    print(f"    --out traj_film_mass_only_0.pkl")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()