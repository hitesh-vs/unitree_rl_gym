import json, os, numpy as np

def variant_difficulty(changes_path):
    with open(changes_path) as f:
        c = json.load(f)
    
    scores = []
    # Mass asymmetry — how different are left/right
    lm = c["group_mass_scale"]["left_leg"]
    rm = c["group_mass_scale"]["right_leg"]
    scores.append(abs(lm - rm))          # asymmetry
    scores.append(abs((lm + rm)/2 - 1))  # deviation from base
    
    # Length deviation
    ll = c["group_length_scale"]["left_leg"]
    rl = c["group_length_scale"]["right_leg"]
    scores.append(abs((ll + rl)/2 - 1))
    
    # Joint range deviation
    jr_vals = list(c["joint_range_scale"].values())
    scores.append(abs(np.mean(jr_vals) - 1))
    
    # Effort deviation  
    ef_vals = list(c["joint_effort_scale"].values())
    scores.append(abs(np.mean(ef_vals) - 1))

    return float(np.mean(scores))

# Score all variants
variant_dir = "resources/robots/g1_variants_wide"
for i in range(10):
    path = f"{variant_dir}/robot_variant_{i}_changes.json"
    score = variant_difficulty(path)
    print(f"robot_variant_{i}: difficulty={score:.3f}")