"""
generate_ood_test_sets.py

Generates 6 OOD test sets as SLIGHT PERTURBATIONS of a reference robot
(variant_4 in this case). Each set varies ONE property slightly while
keeping all other properties exactly as the reference robot.

The idea: these robots are very close to variant_4 which the policy
handles well (882 steps). Being slightly OOD but very similar, the
policy should still generalize — and FiLM should do better than baseline
since it reads the per-limb context.

Perturbation: ±10% noise on the varied property only.

Structure:
    g1_ood_test_sets/
        damping_perturbed/
            variants_metadata.json
            damping_perturbed_robot_0.urdf  ...
        mass_perturbed/  ...
        length_perturbed/  ...
        armature_perturbed/  ...
        joint_range_perturbed/  ...
        all_perturbed/  ...

Usage:
    python generate_ood_test_sets.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_ood_test_sets \
        --base_xml  /path/to/g1_12dof_stripped.xml \
        --seed 7777
"""

import os
import json
import copy
import shutil
import argparse
import numpy as np
import xml.etree.ElementTree as ET

# ── Reference robot (variant_4 exact values) ──────────────────────────────────
REF = {
    "group_mass_scale": {
        "left_leg":  2.4835025360951875,
        "right_leg": 1.6000728820255103,
        "torso":     2.0814278484525035,
    },
    "group_length_scale": {
        "left_leg":  1.077522864531428,
        "right_leg": 1.0984346273702927,
    },
    "joint_range_scale": {
        "left_hip_pitch_joint":   0.6061384407479627,
        "left_hip_roll_joint":    0.7741672315213531,
        "left_hip_yaw_joint":     1.3110361943770594,
        "left_knee_joint":        0.9775329749627818,
        "left_ankle_pitch_joint": 0.9464613494011289,
        "left_ankle_roll_joint":  0.8807195995361908,
        "right_hip_pitch_joint":  1.3625673698628944,
        "right_hip_roll_joint":   1.1937585886195134,
        "right_hip_yaw_joint":    1.0090226823631436,
        "right_knee_joint":       0.6328122788301298,
        "right_ankle_pitch_joint":0.7653264323665646,
        "right_ankle_roll_joint": 0.6342248021411608,
    },
    "joint_effort_scale": {
        "left_hip_pitch_joint":   0.5547846830743501,
        "left_hip_roll_joint":    0.8629187299302165,
        "left_hip_yaw_joint":     0.9896742737581692,
        "left_knee_joint":        1.2766323061344513,
        "left_ankle_pitch_joint": 0.7177725032453276,
        "left_ankle_roll_joint":  1.4003009545989258,
        "right_hip_pitch_joint":  1.1663346907727343,
        "right_hip_roll_joint":   1.1363958146175,
        "right_hip_yaw_joint":    1.4292581394511714,
        "right_knee_joint":       1.058537792313742,
        "right_ankle_pitch_joint":0.9546571572190169,
        "right_ankle_roll_joint": 0.5066842598724905,
    },
    # Damping stored as absolute values (not scales)
    "joint_damping": {
        "left_hip_pitch_joint":   0.001933088634763863,
        "left_hip_roll_joint":    0.0008747039222372677,
        "left_hip_yaw_joint":     0.0019001609092168296,
        "left_knee_joint":        0.0005658530027552562,
        "left_ankle_pitch_joint": 0.001389883706423467,
        "left_ankle_roll_joint":  0.0014429479725339438,
        "right_hip_pitch_joint":  0.0004225353663996092,
        "right_hip_roll_joint":   0.0012773999876451838,
        "right_hip_yaw_joint":    0.0008937097199938902,
        "right_knee_joint":       0.0006244201067998018,
        "right_ankle_pitch_joint":0.0017309959357643102,
        "right_ankle_roll_joint": 0.0006375177539872832,
    },
    # Armature stored as absolute values
    "joint_armature": {
        "left_hip_pitch_joint":   0.005636648873676784,
        "left_hip_roll_joint":    0.005977165743002878,
        "left_hip_yaw_joint":     0.011936588690863429,
        "left_knee_joint":        0.015425784482066534,
        "left_ankle_pitch_joint": 0.01222652482782646,
        "left_ankle_roll_joint":  0.01268820592965347,
        "right_hip_pitch_joint":  0.01704883148732383,
        "right_hip_roll_joint":   0.012234217991781425,
        "right_hip_yaw_joint":    0.01367750035034886,
        "right_knee_joint":       0.006734130292720781,
        "right_ankle_pitch_joint":0.006183394619880478,
        "right_ankle_roll_joint": 0.01219995425665799,
    },
}

LEFT_LEG_LINKS  = {
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
}
RIGHT_LEG_LINKS = {
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
}
TORSO_LINKS     = {"pelvis"}
ACTUATED_JOINTS = set(REF["joint_range_scale"].keys())
BASE_HEIGHT     = 0.78


def perturb(val, rng, pct=0.10):
    """Apply ±pct% noise to a value."""
    return float(val * (1.0 + rng.uniform(-pct, pct)))


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


def apply_full_variant(base_root, mass_scales, range_scales,
                        effort_scales, damping_vals, armature_vals):
    """
    Apply a complete set of property values to the base URDF.
    All values are absolute (not relative to base URDF).
    mass_scales: dict with left_leg, right_leg, torso keys
    range/effort: dict keyed by joint name
    damping/armature: dict keyed by joint name (absolute values)
    """
    root = copy.deepcopy(base_root)

    # ── Mass ─────────────────────────────────────────────────────────────
    for link in root.iter("link"):
        lname    = link.attrib.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            continue

        if lname in LEFT_LEG_LINKS:
            scale = mass_scales["left_leg"]
        elif lname in RIGHT_LEG_LINKS:
            scale = mass_scales["right_leg"]
        elif lname in TORSO_LINKS:
            scale = mass_scales["torso"]
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

    # ── Joint params ──────────────────────────────────────────────────────
    child_to_arm = {}
    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue

        # Range
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            rs   = range_scales[jname]
            lo   = float(limit_elem.attrib.get("lower", "0"))
            hi   = float(limit_elem.attrib.get("upper", "0"))
            ctr  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * rs
            limit_elem.set("lower", f"{ctr - half:.8g}")
            limit_elem.set("upper", f"{ctr + half:.8g}")

            # Effort
            es   = effort_scales[jname]
            orig = float(limit_elem.attrib.get("effort", "88"))
            limit_elem.set("effort", f"{orig * es:.4g}")

        # Damping (absolute value)
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping",  f"{damping_vals[jname]:.6g}")
        dyn.set("friction", "0.0")

        # Armature (absolute value)
        child_elem = joint.find("child")
        if child_elem is not None:
            child_to_arm[child_elem.attrib.get("link", "")] = armature_vals[jname]

    # Apply armature
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


def save_set(set_name, robots, base_urdf_path, base_xml,
             base_leg_z, out_base_dir, base_root):
    """
    Save one set of robots into its own subfolder.
    robots: list of (vroot, changes_dict) tuples
    """
    set_dir = os.path.join(out_base_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)

    base_out = os.path.join(set_dir, "g1_12dof.urdf")
    shutil.copy(base_urdf_path, base_out)

    meta = {
        "g1_12dof": {
            "urdf":                  base_out,
            "xml":                   base_xml,
            "leg_z_left":            round(base_leg_z, 6),
            "leg_z_right":           round(base_leg_z, 6),
            "leg_scale":             1.0,
            "base_height_target":    BASE_HEIGHT,
            "feet_clearance_target": 0.08,
            "set":                   "base",
        }
    }

    for i, (vroot, changes) in enumerate(robots):
        vname    = f"{set_name}_robot_{i}"
        out_urdf = os.path.join(set_dir, f"{vname}.urdf")
        out_json = os.path.join(set_dir, f"{vname}_changes.json")

        indent(vroot)
        ET.ElementTree(vroot).write(
            out_urdf, xml_declaration=True, encoding="unicode")
        with open(out_json, "w") as f:
            json.dump(changes, f, indent=2)

        left_leg_z, right_leg_z = compute_leg_z(vroot)
        avg_leg_z = min(left_leg_z, right_leg_z)
        leg_scale = avg_leg_z / base_leg_z if base_leg_z > 0 else 1.0

        meta[vname] = {
            "urdf":                  out_urdf,
            "xml":                   base_xml,
            "leg_z_left":            round(left_leg_z, 6),
            "leg_z_right":           round(right_leg_z, 6),
            "leg_scale":             round(leg_scale, 6),
            "base_height_target":    round(BASE_HEIGHT * leg_scale, 4),
            "feet_clearance_target": round(0.08 * leg_scale, 4),
            "set":                   set_name,
        }

    meta_path = os.path.join(set_dir, "variants_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved → {set_dir}/  ({len(robots)} robots + base)")
    return meta_path


def make_robot(rng, perturb_prop, pct=0.10):
    """
    Make one robot by perturbing ONLY perturb_prop of the reference robot.
    All other properties are taken EXACTLY from REF.
    perturb_prop: one of 'mass', 'damping', 'armature',
                         'joint_range', 'effort', 'all'
    """
    # Start from reference values exactly
    mass_scales   = copy.deepcopy(REF["group_mass_scale"])
    range_scales  = copy.deepcopy(REF["joint_range_scale"])
    effort_scales = copy.deepcopy(REF["joint_effort_scale"])
    damping_vals  = copy.deepcopy(REF["joint_damping"])
    armature_vals = copy.deepcopy(REF["joint_armature"])

    if perturb_prop in ("mass", "all"):
        for k in mass_scales:
            mass_scales[k] = perturb(mass_scales[k], rng, pct)

    if perturb_prop in ("joint_range", "all"):
        for k in range_scales:
            range_scales[k] = perturb(range_scales[k], rng, pct)

    if perturb_prop in ("effort", "all"):
        for k in effort_scales:
            effort_scales[k] = perturb(effort_scales[k], rng, pct)

    if perturb_prop in ("damping", "all"):
        for k in damping_vals:
            damping_vals[k] = perturb(damping_vals[k], rng, pct)

    if perturb_prop in ("armature", "all"):
        for k in armature_vals:
            armature_vals[k] = perturb(armature_vals[k], rng, pct)

    changes = {
        "reference":       "variant_4",
        "perturbed_prop":  perturb_prop,
        "perturbation_pct": pct,
        "group_mass_scale":   mass_scales,
        "joint_range_scale":  range_scales,
        "joint_effort_scale": effort_scales,
        "joint_damping":      damping_vals,
        "joint_armature":     armature_vals,
    }

    return mass_scales, range_scales, effort_scales, damping_vals, armature_vals, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_urdf",    required=True)
    parser.add_argument("--out_dir",      default="resources/robots/g1_ood_test_sets")
    parser.add_argument("--base_xml",     default="")
    parser.add_argument("--seed",         type=int,   default=7777)
    parser.add_argument("--num_per_set",  type=int,   default=5)
    parser.add_argument("--perturb_pct",  type=float, default=0.10,
                        help="Perturbation percentage (default 10%%)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    N   = args.num_per_set
    PCT = args.perturb_pct

    tree      = ET.parse(args.base_urdf)
    base_root = tree.getroot()

    # Compute base leg Z from unmodified URDF
    base_leg_z = 0.0
    for joint in base_root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname in {"left_hip_pitch_joint", "left_knee_joint",
                     "left_ankle_pitch_joint", "left_ankle_roll_joint"}:
            origin = joint.find("origin")
            if origin is not None:
                xyz = [float(v) for v in
                       origin.attrib.get("xyz", "0 0 0").split()]
                base_leg_z += abs(xyz[2])
    if base_leg_z < 1e-6:
        base_leg_z = 0.1027 + 0.17734 + 0.30001 + 0.017558

    print(f"\n{'='*65}")
    print(f"Generating 6 OOD test sets — slight perturbations of variant_4")
    print(f"Perturbation: ±{PCT*100:.0f}% on varied property only")
    print(f"All other properties: EXACT variant_4 values")
    print(f"Output: {args.out_dir}/")
    print(f"{'='*65}\n")

    sets = [
        ("damping_perturbed",     "damping"),
        ("mass_perturbed",        "mass"),
        ("armature_perturbed",    "armature"),
        ("joint_range_perturbed", "joint_range"),
        ("effort_perturbed",      "effort"),
        ("all_perturbed",         "all"),
    ]

    for set_name, prop in sets:
        print(f"SET: {set_name}  (perturbing '{prop}' by ±{PCT*100:.0f}%)")
        robots = []
        for i in range(N):
            ms, rs, es, dv, av, changes = make_robot(rng, prop, PCT)
            vroot = apply_full_variant(base_root, ms, rs, es, dv, av)
            robots.append((vroot, changes))

            # Print summary for this robot
            if prop == "mass":
                print(f"  robot_{i}: left_mass={ms['left_leg']:.3f}x  "
                      f"right_mass={ms['right_leg']:.3f}x  "
                      f"(ref: 2.484x / 1.600x)")
            elif prop == "damping":
                avg_d = np.mean(list(dv.values()))
                ref_d = np.mean(list(REF["joint_damping"].values()))
                print(f"  robot_{i}: avg_damping={avg_d:.5f}  "
                      f"(ref: {ref_d:.5f})")
            elif prop == "effort":
                avg_e = np.mean(list(es.values()))
                ref_e = np.mean(list(REF["joint_effort_scale"].values()))
                print(f"  robot_{i}: avg_effort={avg_e:.3f}x  "
                      f"(ref: {ref_e:.3f}x)")
            elif prop == "joint_range":
                avg_r = np.mean(list(rs.values()))
                ref_r = np.mean(list(REF["joint_range_scale"].values()))
                print(f"  robot_{i}: avg_range={avg_r:.3f}x  "
                      f"(ref: {ref_r:.3f}x)")
            elif prop == "armature":
                avg_a = np.mean(list(av.values()))
                ref_a = np.mean(list(REF["joint_armature"].values()))
                print(f"  robot_{i}: avg_armature={avg_a:.5f}  "
                      f"(ref: {ref_a:.5f})")
            elif prop == "all":
                print(f"  robot_{i}: all props ±{PCT*100:.0f}% from variant_4")

        save_set(set_name, robots, args.base_urdf, args.base_xml,
                 base_leg_z, args.out_dir, base_root)

    print(f"\n{'='*65}")
    print(f"Done. {6 * N} robots generated across 6 sets.")
    print(f"\nKey property: These robots are variant_4 ± {PCT*100:.0f}%")
    print(f"Policy saw variant_4 during training → should generalize here")
    print(f"\nEval example:")
    print(f"  python record_traj_isaac.py \\")
    print(f"    --checkpoint output_film_wide/.../model_400.pt \\")
    print(f"    --xml_path /path/to/g1_12dof_stripped.xml \\")
    print(f"    --variants_metadata "
          f"{args.out_dir}/mass_perturbed/variants_metadata.json \\")
    print(f"    --variant_name mass_perturbed_robot_0 \\")
    print(f"    --num_eval_rollouts 5 \\")
    print(f"    --out traj_film_mass_perturbed_0.pkl")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()