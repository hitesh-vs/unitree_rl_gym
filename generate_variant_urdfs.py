"""
generate_variant_urdfs.py

Converts MuJoCo XML variants to URDF format for Isaac Gym.
Computes base_height_target per variant by scaling the base robot's
standing height proportionally to the variant's leg length.

Usage:
    python generate_variant_urdfs.py \
        --xml_dir /home/sviswasam/dr/ModuMorph/modular/unitree_g1_actual/xml \
        --base_urdf /home/sviswasam/dr/unitree_rl_gym/resources/robots/g1_description/g1_12dof.urdf \
        --out_dir /home/sviswasam/dr/unitree_rl_gym/resources/robots/g1_variants
"""

import os
import json
import shutil
import argparse
import numpy as np
import xml.etree.ElementTree as ET


# ── Constants from base robot (g1_12dof) ─────────────────────────────────────
# Sum of |z| offsets through kinematic chain: hip_pitch → knee → ankle_pitch → ankle_roll
BASE_LEG_Z      = 0.1027 + 0.17734 + 0.30001 + 0.017558   # = 0.5976
BASE_HEIGHT_TARGET = 0.78   # empirically correct for base robot default pose

# Leg bodies used to compute leg length
LEG_CHAIN = [
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
]


# ── Math helpers ──────────────────────────────────────────────────────────────

def quat_to_rot(q):
    """MuJoCo [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def quat_to_rpy(q):
    """[w,x,y,z] → roll, pitch, yaw (ZYX convention for URDF)."""
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def diag_inertia_to_full(diag, quat):
    """
    MuJoCo diaginertia + orientation quat → URDF inertia tensor components.
    I_body = R @ diag(I) @ R^T
    """
    R = quat_to_rot(quat)
    I = R @ np.diag(diag) @ R.T
    return {
        "ixx": I[0,0], "ixy": I[0,1], "ixz": I[0,2],
        "iyy": I[1,1], "iyz": I[1,2], "izz": I[2,2],
    }


def parse_vec(s):
    return [float(x) for x in s.strip().split()]


# ── Parse MuJoCo XML ──────────────────────────────────────────────────────────

def extract_bodies(xml_path):
    """
    Parse variant XML and return dict of body data.
    Each entry: name → {pos, quat, mass, com, inertia_quat, diaginertia,
                         joint_name, joint_range}
    Also returns leg_z: sum of |z| offsets along LEG_CHAIN.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wb   = root.find("worldbody")

    bodies = {}
    leg_z  = 0.0

    def traverse(elem):
        nonlocal leg_z
        if elem.tag != "body":
            return

        name     = elem.attrib.get("name", "")
        pos      = parse_vec(elem.attrib.get("pos",  "0 0 0"))
        quat     = parse_vec(elem.attrib.get("quat", "1 0 0 0"))

        # Accumulate leg length
        if name in LEG_CHAIN:
            leg_z += abs(pos[2])

        # Inertial
        mass, com, iq, diag = 0.0, [0.,0.,0.], [1.,0.,0.,0.], [1e-6,1e-6,1e-6]
        inertial = elem.find("inertial")
        if inertial is not None:
            mass = float(inertial.attrib.get("mass", "0"))
            com  = parse_vec(inertial.attrib.get("pos",          "0 0 0"))
            iq   = parse_vec(inertial.attrib.get("quat",         "1 0 0 0"))
            diag = parse_vec(inertial.attrib.get("diaginertia",  "1e-6 1e-6 1e-6"))

        # Joint
        jname, jrange = "", [0., 0.]
        joint = elem.find("joint")
        if joint is not None:
            jname  = joint.attrib.get("name", "")
            jrange = parse_vec(joint.attrib.get("range", "0 0"))

        bodies[name] = {
            "pos":          pos,
            "quat":         quat,
            "mass":         mass,
            "com":          com,
            "inertia_quat": iq,
            "diaginertia":  diag,
            "joint_name":   jname,
            "joint_range":  jrange,
        }

        for child in elem.findall("body"):
            traverse(child)

    for body in wb.findall("body"):
        traverse(body)

    return bodies, leg_z


def compute_base_height_target(leg_z):
    """Scale base robot height target proportionally to variant leg length."""
    scale = leg_z / BASE_LEG_Z
    return round(BASE_HEIGHT_TARGET * scale, 4)


# ── Generate URDF from base template + variant body data ─────────────────────

def generate_urdf(base_urdf_path, variant_bodies, out_path):
    tree = ET.parse(base_urdf_path)
    root = tree.getroot()

    # ── Update link inertials ─────────────────────────────────────────────
    for link in root.iter("link"):
        link_name = link.attrib.get("name", "")
        if link_name not in variant_bodies:
            continue
        bd       = variant_bodies[link_name]
        inertial = link.find("inertial")
        if inertial is None:
            continue

        mass_elem = inertial.find("mass")
        if mass_elem is not None:
            mass_elem.set("value", f"{bd['mass']:.8g}")

        origin_elem = inertial.find("origin")
        if origin_elem is not None:
            cx, cy, cz = bd["com"]
            origin_elem.set("xyz", f"{cx:.8g} {cy:.8g} {cz:.8g}")

        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            I = diag_inertia_to_full(bd["diaginertia"], bd["inertia_quat"])
            for k, v in I.items():
                inertia_elem.set(k, f"{v:.8g}")

    # ── Update joint origins and limits ───────────────────────────────────
    for joint in root.iter("joint"):
        jtype = joint.attrib.get("type", "")
        if jtype not in ("revolute", "prismatic", "fixed"):
            continue

        child_elem = joint.find("child")
        if child_elem is None:
            continue
        child_link = child_elem.attrib.get("link", "")
        if child_link not in variant_bodies:
            continue

        bd          = variant_bodies[child_link]
        origin_elem = joint.find("origin")
        if origin_elem is not None:
            px, py, pz = bd["pos"]
            r, p, y    = quat_to_rpy(bd["quat"])
            origin_elem.set("xyz", f"{px:.8g} {py:.8g} {pz:.8g}")
            origin_elem.set("rpy", f"{r:.8g} {p:.8g} {y:.8g}")

        if jtype == "revolute":
            limit_elem = joint.find("limit")
            if limit_elem is not None:
                lo, hi = bd["joint_range"]
                if lo != hi:
                    limit_elem.set("lower", f"{lo:.8g}")
                    limit_elem.set("upper", f"{hi:.8g}")

    _indent(root)
    tree.write(out_path, xml_declaration=True, encoding="unicode")
    print(f"  Written: {out_path}")


def _indent(elem, level=0):
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir",   required=True,
                        help="Dir containing *_stripped.xml files")
    parser.add_argument("--base_urdf", required=True,
                        help="Base URDF (g1_12dof.urdf)")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory for variant URDFs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    xmls = sorted([f for f in os.listdir(args.xml_dir)
                   if f.endswith("_stripped.xml")])

    if not xmls:
        print(f"No *_stripped.xml files found in {args.xml_dir}")
        return

    print(f"Found {len(xmls)} variant XMLs\n")
    metadata = {}

    for xml_file in xmls:
        variant_name = xml_file.replace("_stripped.xml", "")
        xml_path     = os.path.join(args.xml_dir, xml_file)
        print(f"Processing: {variant_name}")

        bodies, leg_z = extract_bodies(xml_path)
        bht           = compute_base_height_target(leg_z)
        out_urdf      = os.path.join(args.out_dir, f"{variant_name}.urdf")

        generate_urdf(args.base_urdf, bodies, out_urdf)

        metadata[variant_name] = {
            "urdf":               out_urdf,
            "xml":                xml_path,
            "leg_z":              round(leg_z, 6),
            "leg_scale":          round(leg_z / BASE_LEG_Z, 6),
            "base_height_target": bht,
            "feet_clearance_target": round(0.08 * (leg_z / BASE_LEG_Z), 4),
        }

        # Copy changes JSON if present
        json_src = os.path.join(args.xml_dir, f"{variant_name}_changes.json")
        if os.path.exists(json_src):
            shutil.copy(json_src,
                        os.path.join(args.out_dir, f"{variant_name}_changes.json"))

    meta_path = os.path.join(args.out_dir, "variants_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata written to: {meta_path}")
    print("\nAll variants:")
    for name, m in metadata.items():
        print(f"  {name}: leg_z={m['leg_z']:.4f}  "
              f"leg_scale={m['leg_scale']:.4f}  "
              f"base_height_target={m['base_height_target']}")


if __name__ == "__main__":
    main()