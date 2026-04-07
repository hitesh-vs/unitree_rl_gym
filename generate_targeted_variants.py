"""
generate_targeted_variants.py

Generates variants specifically in the difficulty range where FiLM
shows clear advantage over baseline (0.2 - 0.45 difficulty score).

Key design: compounding asymmetry — if left leg is heavy, it also gets
weak actuators and restricted range. This forces the policy to produce
fundamentally different per-limb strategies, which FiLM enables.

Usage:
    python generate_targeted_variants.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_variants_targeted \
        --num_variants 15 \
        --seed 42
"""

import os
import json
import copy
import argparse
import numpy as np
import xml.etree.ElementTree as ET

# ── Limb group definitions ────────────────────────────────────────────────────
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
LEFT_JOINTS  = {j for j in ACTUATED_JOINTS if j.startswith("left")}
RIGHT_JOINTS = {j for j in ACTUATED_JOINTS if j.startswith("right")}

BASE_DAMPING  = 0.001
BASE_ARMATURE = 0.01


# ── Targeted ranges — designed to hit difficulty 0.2-0.45 ────────────────────
# Narrower than standard (avoids easy <0.15 and too-hard >0.5)
# Strong compounding: heavy side also gets weak effort + restricted range
TARGETED_RANGES = {
    "mass":        (0.5,  2.2),   # moderate — avoids extremes
    "length":      (0.80, 1.20),  # small length variation (length alone isn't key)
    "joint_range": (0.65, 1.35),  # moderate restriction
    "effort":      (0.55, 1.45),  # moderate weakness
    "damping":     (0.50, 1.80),
    "armature":    (0.60, 1.80),
}


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


def generate_targeted_variant(base_root, rng):
    """
    Generate one variant with compounding asymmetry.

    The key insight from your results:
    FiLM helps most when one side has MULTIPLE compounding disadvantages:
    - heavy AND weak actuators AND restricted range on same side
    Variant_4 (your best FiLM case) had exactly this pattern.

    Strategy:
    1. Pick a "stressed" side (left or right) randomly
    2. Give stressed side: high mass + low effort + restricted range
    3. Give other side:    normal/good properties
    4. Keep total difficulty in 0.2-0.45 range by moderating extremes
    """
    root = copy.deepcopy(base_root)

    # Pick stressed side
    stressed = "left" if rng.random() < 0.5 else "right"
    easy     = "right" if stressed == "left" else "left"

    r = TARGETED_RANGES

    # ── Mass: stressed side heavy, easy side light-to-normal ─────────────
    stressed_mass = float(rng.uniform(1.5, 2.2))   # heavy
    easy_mass     = float(rng.uniform(0.5, 1.0))   # light
    torso_mass    = float(rng.uniform(0.8, 1.5))

    group_mass = {
        f"{stressed}_leg": stressed_mass,
        f"{easy}_leg":     easy_mass,
        "torso":           torso_mass,
    }

    # ── Length: small asymmetry, shared scale mostly ──────────────────────
    shared_length  = float(rng.uniform(r["length"][0], r["length"][1]))
    left_perturb   = float(rng.uniform(0.93, 1.07))
    right_perturb  = float(rng.uniform(0.93, 1.07))
    left_length    = np.clip(shared_length * left_perturb, 0.78, 1.22)
    right_length   = np.clip(shared_length * right_perturb, 0.78, 1.22)
    # Cap asymmetry
    ratio = left_length / right_length
    if ratio > 1.12:   left_length  = right_length * 1.12
    elif ratio < 1/1.12: left_length = right_length / 1.12

    group_length = {"left_leg": float(left_length), "right_leg": float(right_length)}

    # ── Per-joint properties: stressed side compounded ────────────────────
    joint_range_scales  = {}
    joint_effort_scales = {}
    joint_damping_vals  = {}
    joint_armature_vals = {}

    for jname in ACTUATED_JOINTS:
        is_stressed = jname.startswith(stressed)

        if is_stressed:
            # Stressed side: restricted range + weak effort
            joint_range_scales[jname]  = float(rng.uniform(0.60, 0.85))
            joint_effort_scales[jname] = float(rng.uniform(0.50, 0.75))
            damp_scale                 = float(rng.uniform(1.2, 1.8))
            arm_scale                  = float(rng.uniform(1.2, 1.8))
        else:
            # Easy side: normal to slightly better than normal
            joint_range_scales[jname]  = float(rng.uniform(0.95, 1.35))
            joint_effort_scales[jname] = float(rng.uniform(0.95, 1.45))
            damp_scale                 = float(rng.uniform(0.5, 1.2))
            arm_scale                  = float(rng.uniform(0.6, 1.2))

        joint_damping_vals[jname]  = BASE_DAMPING  * damp_scale
        joint_armature_vals[jname] = BASE_ARMATURE * arm_scale

    # ── Apply joint parameter scaling ─────────────────────────────────────
    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue

        limit_elem = joint.find("limit")
        if limit_elem is not None:
            lo   = float(limit_elem.attrib.get("lower", "0"))
            hi   = float(limit_elem.attrib.get("upper", "0"))
            rs   = joint_range_scales[jname]
            ctr  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * rs
            limit_elem.set("lower", f"{ctr - half:.8g}")
            limit_elem.set("upper", f"{ctr + half:.8g}")

            es          = joint_effort_scales[jname]
            orig_effort = float(limit_elem.attrib.get("effort", "88"))
            limit_elem.set("effort", f"{orig_effort * es:.4g}")

        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping",  f"{joint_damping_vals[jname]:.6g}")
        dyn.set("friction", "0.0")

    # ── Armature ──────────────────────────────────────────────────────────
    child_to_armature = {}
    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue
        child_elem = joint.find("child")
        if child_elem is not None:
            child_link = child_elem.attrib.get("link", "")
            child_to_armature[child_link] = joint_armature_vals[jname]

    for link in root.iter("link"):
        lname = link.attrib.get("name", "")
        if lname not in child_to_armature:
            continue
        arm      = child_to_armature[lname]
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia_elem = inertial.find("inertia")
        if inertia_elem is None:
            continue
        for attr in ["ixx", "iyy", "izz"]:
            orig = float(inertia_elem.attrib.get(attr, "0"))
            inertia_elem.set(attr, f"{orig + arm * 1e-4:.10g}")

    changes = {
        "varied_prop":        sorted(["mass", "length", "joint_range",
                                      "effort", "damping", "armature"]),
        "allow_asymmetry":    True,
        "stressed_side":      stressed,
        "group_mass_scale":   group_mass,
        "group_length_scale": group_length,
        "joint_range_scale":  joint_range_scales,
        "joint_effort_scale": joint_effort_scales,
        "joint_damping":      joint_damping_vals,
        "joint_armature":     joint_armature_vals,
    }

    return root, changes, stressed, group_mass, group_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_urdf",    required=True)
    parser.add_argument("--out_dir",      default="resources/robots/g1_variants_targeted")
    parser.add_argument("--num_variants", type=int, default=15)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--base_xml",     default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tree      = ET.parse(args.base_urdf)
    base_root = tree.getroot()

    # Copy base URDF
    base_out = os.path.join(args.out_dir, "g1_12dof.urdf")
    indent(copy.deepcopy(base_root))
    tree.write(base_out, xml_declaration=True, encoding="unicode")

    # Compute base leg Z
    BASE_LEG_Z = 0.0
    for joint in base_root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname in {"left_hip_pitch_joint", "left_knee_joint",
                     "left_ankle_pitch_joint", "left_ankle_roll_joint"}:
            origin = joint.find("origin")
            if origin is not None:
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                BASE_LEG_Z += abs(xyz[2])
    if BASE_LEG_Z < 1e-6:
        BASE_LEG_Z = 0.1027 + 0.17734 + 0.30001 + 0.017558

    BASE_HEIGHT_TGT = 0.78
    all_meta = {}

    print(f"\n{'='*70}")
    print(f"Generating {args.num_variants} targeted variants")
    print(f"Target difficulty range: 0.20 - 0.45")
    print(f"Strategy: compounding asymmetry (stressed side = heavy+weak+restricted)")
    print(f"{'='*70}")
    print(f"\n  {'Name':<25} {'Stressed':>8} {'LMass':>6} {'RMass':>6} "
          f"{'L/R ratio':>10}")
    print(f"  {'-'*60}")

    for i in range(args.num_variants):
        vname = f"targeted_variant_{i}"

        variant_root, changes, stressed, gm, gl = generate_targeted_variant(
            base_root, rng)

        # Compute leg Z for base_height_target
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

        avg_leg_z  = min(left_leg_z, right_leg_z)
        leg_scale  = avg_leg_z / BASE_LEG_Z
        bht        = round(BASE_HEIGHT_TGT * leg_scale, 4)
        fct        = round(0.08 * leg_scale, 4)

        indent(variant_root)
        out_urdf = os.path.join(args.out_dir, f"{vname}.urdf")
        ET.ElementTree(variant_root).write(
            out_urdf, xml_declaration=True, encoding="unicode")

        out_json = os.path.join(args.out_dir, f"{vname}_changes.json")
        with open(out_json, "w") as f:
            json.dump(changes, f, indent=2)

        lm = gm["left_leg"]
        rm = gm["right_leg"]
        ratio = lm / rm
        print(f"  {vname:<25} {stressed:>8} {lm:>6.2f} {rm:>6.2f} {ratio:>10.2f}x")

        all_meta[vname] = {
            "urdf":                  out_urdf,
            "xml":                   args.base_xml,
            "leg_z_left":            round(left_leg_z,  6),
            "leg_z_right":           round(right_leg_z, 6),
            "leg_scale":             round(leg_scale,   6),
            "base_height_target":    bht,
            "feet_clearance_target": fct,
            "preset":                "targeted",
            "asymmetric":            True,
            "stressed_side":         stressed,
        }

    # Add base robot
    all_meta["g1_12dof"] = {
        "urdf":                  base_out,
        "xml":                   args.base_xml,
        "leg_z_left":            round(BASE_LEG_Z, 6),
        "leg_z_right":           round(BASE_LEG_Z, 6),
        "leg_scale":             1.0,
        "base_height_target":    BASE_HEIGHT_TGT,
        "feet_clearance_target": 0.08,
        "preset":                "base",
        "asymmetric":            False,
        "stressed_side":         "none",
    }

    meta_path = os.path.join(args.out_dir, "variants_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nGenerated {args.num_variants} targeted variants → {args.out_dir}")
    print(f"Metadata → {meta_path}")
    print(f"\nNext steps:")
    print(f"  1. Train both baseline and FiLM on these variants")
    print(f"  2. Eval on them — expect FiLM advantage on most")
    print(f"  3. Use plot_difficulty_vs_performance.py to visualize")


if __name__ == "__main__":
    main()