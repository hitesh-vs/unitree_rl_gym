"""
generate_urdf_variants.py

Generates URDF robot variants with rich parameter variation:
  - mass:        per limb-group (left_leg, right_leg, torso) independent scale
  - length:      per limb-group, leg chain joints only
  - joint_range: per joint independent scale
  - effort:      per joint (URDF equivalent of gear ratio)
  - damping:     per joint via <dynamics> element
  - armature:    encoded as small added diagonal inertia (no URDF native support)

Limb groups:
  left_leg:  left_hip_pitch, left_hip_roll, left_hip_yaw,
             left_knee, left_ankle_pitch, left_ankle_roll
  right_leg: right_hip_pitch, right_hip_roll, right_hip_yaw,
             right_knee, right_ankle_pitch, right_ankle_roll
  torso:     pelvis, torso_link (and all fixed children)

Usage:
    python generate_urdf_variants.py \
        --base_urdf resources/robots/g1_description/g1_12dof.urdf \
        --out_dir   resources/robots/g1_variants_v2 \
        --num_variants 10 \
        --seed 2025
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
TORSO_LINKS = {"pelvis"}   # pelvis only — torso_link is fixed, not actuated

# Joints whose parent joint origin (xyz) gets scaled for leg length
# Only the z-component of the structural chain matters
LEG_LENGTH_JOINTS = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
}

# Actuated joints (revolute) — gear/damping/range vary for these only
ACTUATED_JOINTS = LEG_LENGTH_JOINTS  # same set for G1 12-DOF

# ── Randomisation ranges ──────────────────────────────────────────────────────

RANGES = {
    "mass":        (0.7, 1.4),   # per limb-group scale
    "length":      (0.85, 1.15), # per limb-group scale (leg chain z-offsets)
    "joint_range": (0.90, 1.10), # per joint, symmetric around centre
    "effort":      (0.80, 1.20), # per joint (gear proxy)
    "damping":     (0.70, 1.30), # per joint, applied to <dynamics>
    "armature":    (0.80, 1.20), # per joint, encoded as added inertia diagonal
}

# Base damping value (URDF has no default — we add it explicitly)
BASE_DAMPING  = 0.001
BASE_ARMATURE = 0.01   # encoded as ixx=iyy=izz += armature * base_inertia_norm


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample(rng, key):
    lo, hi = RANGES[key]
    return float(rng.uniform(lo, hi))


def scale_inertia(inertia_elem, mass_scale):
    """
    Scale all inertia tensor components by mass_scale.
    Inertia ~ mass * length^2, so when mass scales, inertia scales proportionally.
    We use mass_scale (not mass_scale * length_scale^2) to keep it decoupled.
    """
    attrs = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
    for a in attrs:
        if a in inertia_elem.attrib:
            inertia_elem.set(a, f"{float(inertia_elem.attrib[a]) * mass_scale:.10g}")


def get_link_group(link_name):
    if link_name in LEFT_LEG_LINKS:  return "left_leg"
    if link_name in RIGHT_LEG_LINKS: return "right_leg"
    if link_name in TORSO_LINKS:     return "torso"
    return None


# ── Variant generator ─────────────────────────────────────────────────────────

def generate_variant(base_root, rng, variant_idx):
    """
    Generate one URDF variant. Returns (root_elem, changes_dict).
    base_root is NOT modified — we deepcopy it first.
    """
    root = copy.deepcopy(base_root)

    # ── Sample group-level scales ─────────────────────────────────────────
    group_mass   = {g: sample(rng, "mass")   for g in ["left_leg", "right_leg", "torso"]}
    group_length = {g: sample(rng, "length") for g in ["left_leg", "right_leg"]}
    # torso length not varied — pelvis is the root, changing its z offset
    # would shift the whole robot spawn height unpredictably

    # ── Sample per-joint scales ───────────────────────────────────────────
    joint_range_scales  = {}
    joint_effort_scales = {}
    joint_damping_vals  = {}
    joint_armature_vals = {}

    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue
        joint_range_scales[jname]  = sample(rng, "joint_range")
        joint_effort_scales[jname] = sample(rng, "effort")
        joint_damping_vals[jname]  = BASE_DAMPING * sample(rng, "damping")
        joint_armature_vals[jname] = BASE_ARMATURE * sample(rng, "armature")

    # ── Apply mass scaling ────────────────────────────────────────────────
    for link in root.iter("link"):
        lname = link.attrib.get("name", "")
        grp   = get_link_group(lname)
        if grp is None:
            continue

        ms       = group_mass[grp]
        inertial = link.find("inertial")
        if inertial is None:
            continue

        mass_elem = inertial.find("mass")
        if mass_elem is not None:
            orig = float(mass_elem.attrib["value"])
            mass_elem.set("value", f"{orig * ms:.8g}")

        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            scale_inertia(inertia_elem, ms)

    # ── Apply length scaling (leg chain joint origins) ────────────────────
    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in LEG_LENGTH_JOINTS:
            continue

        # Determine which group this joint belongs to
        grp = ("left_leg"  if jname.startswith("left")  else
               "right_leg" if jname.startswith("right") else None)
        if grp is None:
            continue

        ls          = group_length[grp]
        origin_elem = joint.find("origin")
        if origin_elem is None:
            continue

        xyz_str = origin_elem.attrib.get("xyz", "0 0 0")
        xyz     = [float(v) for v in xyz_str.split()]
        # Scale only the z component (structural length along leg chain)
        xyz[2]  = xyz[2] * ls
        origin_elem.set("xyz", f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g}")

    # ── Apply joint parameter scaling ─────────────────────────────────────
    for joint in root.iter("joint"):
        jname = joint.attrib.get("name", "")
        if jname not in ACTUATED_JOINTS:
            continue

        # Joint range — scale symmetrically around centre
        limit_elem = joint.find("limit")
        if limit_elem is not None:
            lo   = float(limit_elem.attrib.get("lower", "0"))
            hi   = float(limit_elem.attrib.get("upper", "0"))
            rs   = joint_range_scales[jname]
            ctr  = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * rs
            limit_elem.set("lower", f"{ctr - half:.8g}")
            limit_elem.set("upper", f"{ctr + half:.8g}")

            # Effort (gear proxy)
            es   = joint_effort_scales[jname]
            orig_effort = float(limit_elem.attrib.get("effort", "88"))
            limit_elem.set("effort", f"{orig_effort * es:.4g}")

        # Damping — add/update <dynamics> element
        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping",  f"{joint_damping_vals[jname]:.6g}")
        dyn.set("friction", "0.0")   # keep friction at 0, only vary damping

    # ── Armature: encode as small diagonal inertia addition ───────────────
    # URDF has no armature field. We add armature * 1e-4 to ixx=iyy=izz
    # of the child link for each actuated joint. This approximates the
    # rotational inertia of the motor rotor.
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

    # ── Build changes dict ────────────────────────────────────────────────
    changes = {
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
    """Pretty-print indentation."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_urdf",    required=True)
    parser.add_argument("--out_dir",      default="resources/robots/g1_variants_v2")
    parser.add_argument("--num_variants", type=int, default=10)
    parser.add_argument("--seed",         type=int, default=2025,
                        help="Seed for reproducibility (default: 2025)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Parse base URDF once
    tree      = ET.parse(args.base_urdf)
    base_root = tree.getroot()

    # Also copy base URDF as variant 0 (identity) for reference
    base_out = os.path.join(args.out_dir, "g1_12dof.urdf")
    indent(copy.deepcopy(base_root))
    tree.write(base_out, xml_declaration=True, encoding="unicode")
    print(f"[base] Copied → {base_out}")

    rng = np.random.default_rng(args.seed)
    all_meta = {}

    # Base height target for base robot
    BASE_LEG_Z      = 0.1027 + 0.17734 + 0.30001 + 0.017558
    BASE_HEIGHT_TGT = 0.78

    for i in range(args.num_variants):
        variant_name = f"robot_variant_{i}"
        print(f"\nGenerating {variant_name} ...")

        variant_root, changes = generate_variant(base_root, rng, i)

        # Compute leg_z for base_height_target
        left_leg_z = 0.0
        for joint in variant_root.iter("joint"):
            jname = joint.attrib.get("name", "")
            if jname not in {
                "left_hip_pitch_joint", "left_knee_joint",
                "left_ankle_pitch_joint", "left_ankle_roll_joint",
            }:
                continue
            origin = joint.find("origin")
            if origin is not None:
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                left_leg_z += abs(xyz[2])

        leg_scale   = left_leg_z / BASE_LEG_Z
        bht         = round(BASE_HEIGHT_TGT * leg_scale, 4)
        fct         = round(0.08 * leg_scale, 4)

        # Save URDF
        indent(variant_root)
        out_urdf = os.path.join(args.out_dir, f"{variant_name}.urdf")
        ET.ElementTree(variant_root).write(
            out_urdf, xml_declaration=True, encoding="unicode")

        # Save changes JSON
        out_json = os.path.join(args.out_dir, f"{variant_name}_changes.json")
        with open(out_json, "w") as f:
            json.dump(changes, f, indent=2)

        all_meta[variant_name] = {
            "urdf":                   out_urdf,
            "xml":                    out_urdf,   # runner expects "xml" key too
            "leg_z":                  round(left_leg_z, 6),
            "leg_scale":              round(leg_scale, 6),
            "base_height_target":     bht,
            "feet_clearance_target":  fct,
        }

        print(f"  leg_scale={leg_scale:.4f}  base_height_target={bht}  "
              f"effort_scales={list(changes['joint_effort_scale'].values())[:3]}...")
        print(f"  Written → {out_urdf}")

    # Add base robot to metadata
    all_meta["g1_12dof"] = {
        "urdf":                   base_out,
        "xml":                    base_out,
        "leg_z":                  round(BASE_LEG_Z, 6),
        "leg_scale":              1.0,
        "base_height_target":     BASE_HEIGHT_TGT,
        "feet_clearance_target":  0.08,
    }

    # Write combined metadata
    meta_path = os.path.join(args.out_dir, "variants_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {args.num_variants} variants + base robot")
    print(f"Metadata → {meta_path}")
    print(f"Seed used: {args.seed}  (use same seed to reproduce)")
    print(f"{'='*60}")
    print("\nVariant summary:")
    print(f"  {'Name':<22} {'LegScale':>9} {'BaseHt':>7} {'FeetClear':>10}")
    print(f"  {'-'*52}")
    for name, m in all_meta.items():
        print(f"  {name:<22} {m['leg_scale']:>9.4f} "
              f"{m['base_height_target']:>7.4f} "
              f"{m['feet_clearance_target']:>10.4f}")


if __name__ == "__main__":
    main()