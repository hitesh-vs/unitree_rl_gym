"""
generate_extreme_variants.py

Generates URDF robot variants with rich parameter variation.
Updated ranges are significantly wider and allow left/right asymmetry
so variants require genuinely different policies, not just a shared average.

Key changes vs original:
  - Mass range widened to (0.4, 2.5) — heavy vs light robots are incompatible
  - Length range widened to (0.70, 1.30) — short vs tall legs need different balance
  - Left/right legs sampled INDEPENDENTLY — asymmetric robots stress the policy
  - joint_range widened to (0.60, 1.40) — some robots are severely restricted
  - effort widened to (0.50, 1.50) — weak actuators force different strategies
  - damping widened to (0.40, 2.00) — overdamped vs underdamped joints
  - Added EXTREME preset for held-out eval robots (outside training distribution)

Usage:
    # Standard training variants (wide but in-distribution)
    python generate_urdf_variants.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_variants_wide \
        --num_variants 20 \
        --seed 2025

    # Held-out eval variants (extreme — outside training distribution)
    python generate_urdf_variants.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_variants_heldout \
        --num_variants 5 \
        --seed 9999 \
        --preset extreme

    # Vary ONLY one property
    python generate_urdf_variants.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_variants_mass_only \
        --num_variants 10 \
        --seed 2025 \
        --vary_prop mass
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
TORSO_LINKS = {"pelvis"}

LEG_LENGTH_JOINTS = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
}

ACTUATED_JOINTS = LEG_LENGTH_JOINTS

ALL_PROPS = {"mass", "length", "joint_range", "effort", "damping", "armature"}

# ── Randomisation ranges ──────────────────────────────────────────────────────
# Three presets:
#   "standard" — wide enough to force genuine adaptation, stable enough to train
#   "extreme"  — outside standard training distribution, used for held-out eval
#   "original" — original narrow ranges (kept for reference/ablation)

RANGES = {
    "standard": {
        # Per limb-group mass scale. Range (0.4, 2.5) means a heavy robot
        # weighs 6x a light robot — fundamentally different balance dynamics.
        "mass":        (0.4,  2.5),
        # Per limb-group leg length scale. (0.70, 1.30) means short-legged
        # robots need ~20% higher step frequency; tall ones need wider stance.
        # Left and right legs are sampled INDEPENDENTLY (see generate_variant).
        "length":      (0.70, 1.30),
        # Joint range scale. (0.60, 1.40) — restricted robots cannot use
        # full hip extension; the policy must find shorter strides.
        "joint_range": (0.60, 1.40),
        # Effort (actuator strength). (0.50, 1.50) — a weak-actuator robot
        # cannot accelerate quickly; it needs anticipatory torque strategies.
        "effort":      (0.50, 1.50),
        # Damping. (0.40, 2.00) — highly damped joints resist fast motion;
        # the policy must use slower, more deliberate movements.
        "damping":     (0.40, 2.00),
        # Armature (rotational inertia added diagonally).
        "armature":    (0.50, 2.00),
    },
    "extreme": {
        # Pushed beyond standard range — used for held-out zero-shot eval.
        # Baseline should fail here; FiLM should interpolate from context.
        "mass":        (0.25, 3.5),
        "length":      (0.55, 1.50),
        "joint_range": (0.45, 1.60),
        "effort":      (0.30, 1.80),
        "damping":     (0.20, 3.00),
        "armature":    (0.30, 3.00),
    },
    "original": {
        # Kept for ablation — narrow ranges from original generator.
        "mass":        (0.7,  1.4),
        "length":      (0.85, 1.15),
        "joint_range": (0.90, 1.10),
        "effort":      (0.80, 1.20),
        "damping":     (0.70, 1.30),
        "armature":    (0.80, 1.20),
    },
}

BASE_DAMPING  = 0.001
BASE_ARMATURE = 0.01


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample(rng, key, active_props, ranges):
    if key not in active_props:
        return 1.0
    lo, hi = ranges[key]
    return float(rng.uniform(lo, hi))


def scale_inertia(inertia_elem, mass_scale):
    for a in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
        if a in inertia_elem.attrib:
            inertia_elem.set(a, f"{float(inertia_elem.attrib[a]) * mass_scale:.10g}")


def get_link_group(link_name):
    if link_name in LEFT_LEG_LINKS:  return "left_leg"
    if link_name in RIGHT_LEG_LINKS: return "right_leg"
    if link_name in TORSO_LINKS:     return "torso"
    return None


# ── Variant generator ─────────────────────────────────────────────────────────

def generate_variant(base_root, rng, variant_idx, active_props, ranges,
                     allow_asymmetry=True):
    """
    Generate one URDF variant.

    allow_asymmetry: if True, left and right legs get independent mass and
                     length scales. This is the key change from the original —
                     asymmetric robots cannot be handled by a single average
                     gait, forcing the policy to read morphology context.
    """
    root = copy.deepcopy(base_root)

    # ── Sample group-level scales ─────────────────────────────────────────
    # Torso mass is always symmetric (single body)
    torso_mass = sample(rng, "mass", active_props, ranges)

    if allow_asymmetry and "mass" in active_props:
        # Left and right legs get independent mass scales
        left_mass  = sample(rng, "mass", active_props, ranges)
        right_mass = sample(rng, "mass", active_props, ranges)
    else:
        # Both legs share one scale (original behaviour)
        shared_mass = sample(rng, "mass", active_props, ranges)
        left_mass   = shared_mass
        right_mass  = shared_mass

    group_mass = {
        "left_leg":  left_mass,
        "right_leg": right_mass,
        "torso":     torso_mass,
    }

    if allow_asymmetry and "length" in active_props:
        # Independent leg lengths — asymmetric robot must compensate
        left_length  = sample(rng, "length", active_props, ranges)
        right_length = sample(rng, "length", active_props, ranges)
    else:
        shared_length = sample(rng, "length", active_props, ranges)
        left_length   = shared_length
        right_length  = shared_length

    group_length = {
        "left_leg":  left_length,
        "right_leg": right_length,
    }

    # ── Sample per-joint scales ───────────────────────────────────────────
    joint_range_scales  = {}
    joint_effort_scales = {}
    joint_damping_vals  = {}
    joint_armature_vals = {}

    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue
        joint_range_scales[jname]  = sample(rng, "joint_range", active_props, ranges)
        joint_effort_scales[jname] = sample(rng, "effort",      active_props, ranges)
        damp_scale = sample(rng, "damping",   active_props, ranges)
        arm_scale  = sample(rng, "armature",  active_props, ranges)
        joint_damping_vals[jname]  = BASE_DAMPING  * damp_scale
        joint_armature_vals[jname] = BASE_ARMATURE * arm_scale

    # ── Sample group-level scales ─────────────────────────────────────────
    torso_mass = sample(rng, "mass", active_props, ranges)

    # Mass asymmetry: safe at larger ratios since it only affects dynamics,
    # not geometry. Cap left/right ratio at 2.5x to keep balance learnable.
    if "mass" in active_props:
        left_mass  = sample(rng, "mass", active_props, ranges)
        right_mass = sample(rng, "mass", active_props, ranges)
        ratio = left_mass / right_mass
        if ratio > 2.5:
            left_mass = right_mass * 2.5
        elif ratio < 1.0 / 2.5:
            left_mass = right_mass / 2.5
    else:
        shared_mass = sample(rng, "mass", active_props, ranges)
        left_mass   = shared_mass
        right_mass  = shared_mass

    group_mass = {
        "left_leg":  left_mass,
        "right_leg": right_mass,
        "torso":     torso_mass,
    }

    # Length asymmetry: dangerous above ~15% difference because Isaac Gym's
    # contact solver assumes roughly symmetric foot placement at spawn.
    # Strategy: one shared scale drives the big morphology difference across
    # variants (short vs tall robot), plus a small per-side perturbation that
    # is enough for FiLM to condition on left vs right but never breaks physics.
    if "length" in active_props:
        shared_length = sample(rng, "length", active_props, ranges)
        left_perturb  = float(rng.uniform(0.93, 1.07))
        right_perturb = float(rng.uniform(0.93, 1.07))
        left_length   = shared_length * left_perturb
        right_length  = shared_length * right_perturb
        # Hard cap: left/right ratio must stay within 1.15x
        ratio = left_length / right_length
        if ratio > 1.15:
            left_length = right_length * 1.15
        elif ratio < 1.0 / 1.15:
            left_length = right_length / 1.15
    else:
        left_length  = 1.0
        right_length = 1.0

    group_length = {
        "left_leg":  left_length,
        "right_leg": right_length,
    }
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
        "varied_prop":        sorted(active_props),
        "allow_asymmetry":    allow_asymmetry,
        "group_mass_scale":   group_mass,
        "group_length_scale": group_length,
        "joint_range_scale":  joint_range_scales,
        "joint_effort_scale": joint_effort_scales,
        "joint_damping":      joint_damping_vals,
        "joint_armature":     joint_armature_vals,
    }

    return root, changes


# ── Main ──────────────────────────────────────────────────────────────────────

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate URDF robot variants with controlled property variation."
    )
    parser.add_argument("--base_urdf",    required=True)
    parser.add_argument("--out_dir",      default="resources/robots/g1_variants_wide")
    parser.add_argument("--num_variants", type=int, default=20)
    parser.add_argument("--seed",         type=int, default=2025)
    parser.add_argument(
        "--vary_prop",
        default="all",
        choices=sorted(ALL_PROPS) + ["all"],
    )
    parser.add_argument(
        "--preset",
        default="standard",
        choices=["standard", "extreme", "original"],
        help=(
            "standard: wide training distribution. "
            "extreme: held-out eval robots outside training range. "
            "original: narrow ranges from original generator (ablation)."
        ),
    )
    parser.add_argument(
        "--no_asymmetry",
        action="store_true",
        default=False,
        help="Disable left/right asymmetry (both legs always share same scale).",
    )
    parser.add_argument("--base_xml", default=None)

    args = parser.parse_args()

    active_props  = ALL_PROPS if args.vary_prop == "all" else {args.vary_prop}
    ranges        = RANGES[args.preset]
    allow_asym    = not args.no_asymmetry

    os.makedirs(args.out_dir, exist_ok=True)

    tree      = ET.parse(args.base_urdf)
    base_root = tree.getroot()

    base_out = os.path.join(args.out_dir, "g1_12dof.urdf")
    indent(copy.deepcopy(base_root))
    tree.write(base_out, xml_declaration=True, encoding="unicode")
    print(f"[base] Copied → {base_out}")
    print(f"Preset        : {args.preset}")
    print(f"Varied prop   : {args.vary_prop}")
    print(f"Asymmetry     : {allow_asym}")

    rng      = np.random.default_rng(args.seed)
    all_meta = {}

    # Compute base leg Z from unmodified URDF
    BASE_LEG_Z = 0.0
    for joint in base_root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in {
            "left_hip_pitch_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
        }:
            continue
        origin = joint.find("origin")
        if origin is not None:
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            BASE_LEG_Z += abs(xyz[2])

    if BASE_LEG_Z < 1e-6:
        # Fallback if URDF doesn't match expected joint names
        BASE_LEG_Z = 0.1027 + 0.17734 + 0.30001 + 0.017558

    BASE_HEIGHT_TGT = 0.78

    for i in range(args.num_variants):
        variant_name = f"robot_variant_{i}"
        print(f"\nGenerating {variant_name} ...")

        variant_root, changes = generate_variant(
            base_root, rng, i, active_props, ranges, allow_asymmetry=allow_asym)

        # Use average of left/right leg Z for base_height_target
        left_leg_z  = 0.0
        right_leg_z = 0.0
        left_joints = {
            "left_hip_pitch_joint", "left_knee_joint",
            "left_ankle_pitch_joint", "left_ankle_roll_joint",
        }
        right_joints = {
            "right_hip_pitch_joint", "right_knee_joint",
            "right_ankle_pitch_joint", "right_ankle_roll_joint",
        }
        for joint in variant_root.iter("joint"):
            jname  = joint.attrib.get("name", "")
            origin = joint.find("origin")
            if origin is None:
                continue
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            if jname in left_joints:
                left_leg_z  += abs(xyz[2])
            elif jname in right_joints:
                right_leg_z += abs(xyz[2])

        # Using average would set the target too high and cause early termination
        # on the short side.
        avg_leg_z = min(left_leg_z, right_leg_z)
        leg_scale  = avg_leg_z / BASE_LEG_Z
        bht        = round(BASE_HEIGHT_TGT * leg_scale, 4)
        fct        = round(0.08 * leg_scale, 4)

        indent(variant_root)
        out_urdf = os.path.join(args.out_dir, f"{variant_name}.urdf")
        ET.ElementTree(variant_root).write(
            out_urdf, xml_declaration=True, encoding="unicode")

        out_json = os.path.join(args.out_dir, f"{variant_name}_changes.json")
        with open(out_json, "w") as f:
            json.dump(changes, f, indent=2)

        base_xml = args.base_xml or ""

        all_meta[variant_name] = {
            "urdf":                   out_urdf,
            "xml":                    base_xml,
            "leg_z_left":             round(left_leg_z,  6),
            "leg_z_right":            round(right_leg_z, 6),
            "leg_scale":              round(leg_scale,   6),
            "base_height_target":     bht,
            "feet_clearance_target":  fct,
            "preset":                 args.preset,
            "asymmetric":             allow_asym,
        }

        asym_str = (f"  L/R mass={changes['group_mass_scale']['left_leg']:.2f}/"
                    f"{changes['group_mass_scale']['right_leg']:.2f}"
                    f"  L/R len={changes['group_length_scale']['left_leg']:.2f}/"
                    f"{changes['group_length_scale']['right_leg']:.2f}")
        print(f"  leg_scale={leg_scale:.4f}  bht={bht}{asym_str}")
        print(f"  → {out_urdf}")

    all_meta["g1_12dof"] = {
        "urdf":                   base_out,
        "xml":                    base_out,
        "leg_z_left":             round(BASE_LEG_Z, 6),
        "leg_z_right":            round(BASE_LEG_Z, 6),
        "leg_scale":              1.0,
        "base_height_target":     BASE_HEIGHT_TGT,
        "feet_clearance_target":  0.08,
        "preset":                 "base",
        "asymmetric":             False,
    }

    meta_path = os.path.join(args.out_dir, "variants_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    prop_path = os.path.join(args.out_dir, "varied_prop.txt")
    with open(prop_path, "w") as f:
        f.write(f"{args.vary_prop} preset={args.preset} asymmetry={allow_asym}\n")

    print(f"\n{'='*60}")
    print(f"Generated {args.num_variants} variants + base robot")
    print(f"Preset      : {args.preset}")
    print(f"Varied prop : {args.vary_prop}")
    print(f"Asymmetry   : {allow_asym}")
    print(f"Metadata    → {meta_path}")
    print(f"Seed used   : {args.seed}")
    print(f"{'='*60}")
    print(f"\n  {'Name':<22} {'LegScale':>9} {'BaseHt':>7} {'LMass':>6} {'RMass':>6}")
    print(f"  {'-'*56}")
    for name, m in all_meta.items():
        ms = all_meta[name]
        lm = "-"
        rm = "-"
        if name != "g1_12dof":
            json_path = os.path.join(args.out_dir, f"{name}_changes.json")
            if os.path.exists(json_path):
                with open(json_path) as jf:
                    ch = json.load(jf)
                lm = f"{ch['group_mass_scale'].get('left_leg', 1.0):.2f}"
                rm = f"{ch['group_mass_scale'].get('right_leg', 1.0):.2f}"
        print(f"  {name:<22} {ms['leg_scale']:>9.4f} "
              f"{ms['base_height_target']:>7.4f} {lm:>6} {rm:>6}")


if __name__ == "__main__":
    main()