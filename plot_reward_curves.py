# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# # --- CONFIGURATION ---
# LOG_DIR = 'output_baseline_results2/Apr11_18-52-43/events.out.tfevents.1775947991.gpu-5-26.int.turing.wpi.edu.1252508.0' # Directory containing the tfevents file
# OUTPUT_NAME = 'eplen_variants.pdf'
# SMOOTHING_WEIGHT = 0.8  # Adjust 0.0 to 1.0 (higher is smoother)

# def smooth(values, weight):
#     last = values[0]
#     smoothed = []
#     for point in values:
#         sampled_avg = last * weight + (1 - weight) * point
#         smoothed.append(sampled_avg)
#         last = sampled_avg
#     return smoothed

# # 1. Initialize EventAccumulator
# # We set size_guidance to 0 to get ALL samples (no downsampling)
# event_acc = EventAccumulator(LOG_DIR, size_guidance={'scalars': 0})
# event_acc.Reload()

# # 2. Extract specific tags
# all_tags = event_acc.Tags()['scalars']
# target_tags = [t for t in all_tags if 'EpLen' in t]
# target_tags.sort() # Ensures variant_0 comes before variant_1, etc.

# # 3. Setup Plotting Aesthetics
# plt.style.use('seaborn-v0_8-paper') # Clean, professional base
# plt.rcParams.update({
#     "text.usetex": False,            # Set to True if you have LaTeX installed
#     "font.family": "serif",
#     "axes.spines.top": False,
#     "axes.spines.right": False,
# })

# fig, ax = plt.subplots(figsize=(8, 5))
# cmap = plt.get_cmap('tab10') # Distinct colors for 10 variants

# # 4. Loop through tags and plot
# for i, tag in enumerate(target_tags):
#     events = event_acc.Scalars(tag)
#     steps = [e.step for e in events]
#     values = [e.value for e in events]
    
#     if len(values) == 0:
#         continue
        
#     # Smooth data for clarity
#     smoothed_vals = smooth(values, SMOOTHING_WEIGHT)
    
#     # Extract short name for legend (e.g., "variant_0")
#     label = tag.split('/')[-1]
    
#     # Plot raw data (faded) and smoothed data (bold)
#     line, = ax.plot(steps, smoothed_vals, label=label, color=cmap(i), linewidth=2)
#     ax.plot(steps, values, color=line.get_color(), alpha=0.15, linewidth=0.8)

# # 5. Final Formatting
# ax.set_title('Episode Length Across Variants', fontsize=14, pad=15)
# ax.set_xlabel('Global Steps', fontsize=12)
# ax.set_ylabel('Episode Length', fontsize=12)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
# ax.grid(alpha=0.3, linestyle='--')

# plt.tight_layout()
# plt.savefig(OUTPUT_NAME, dpi=300)
# print(f"Successfully saved plot to {OUTPUT_NAME}")
# plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
FILM_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_film_wide/Mar31_18-18-17/events.out.tfevents.1774995525.gpu-5-25.int.turing.wpi.edu.2692739.0'
BASELINE_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_baseline_wide/Mar31_18-20-20/events.out.tfevents.1774995648.gpu-5-23.int.turing.wpi.edu.2051735.0'
TARGET_TAGS = ['EpLen/variant_robot_variant_4', 'EpLen/variant_robot_variant_6']
BASELINE_CAP = 420  # Cap baseline at this iteration
SMOOTHING = 0.85

def get_averaged_data(log_dir, tags, cap_step=None):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    all_series = []
    global_max_step = 0
    global_min_step = float('inf')
    
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        
        # Apply the cap if specified
        if cap_step is not None:
            mask = steps <= cap_step
            steps = steps[mask]
            values = values[mask]
            
        if len(steps) == 0: continue
        
        all_series.append((steps, values))
        global_max_step = max(global_max_step, steps.max())
        global_min_step = min(global_min_step, steps.min())

    # Create common axis
    common_steps = np.linspace(global_min_step, global_max_step, 1000)
    interp_values = [np.interp(common_steps, s, v) for s, v in all_series]
    
    return common_steps, np.mean(interp_values, axis=0)

def smooth(values, weight):
    last = values[0]
    smoothed = []
    for point in values:
        sampled_avg = last * weight + (1 - weight) * point
        smoothed.append(sampled_avg)
        last = smoothed[-1]
    return np.array(smoothed)

# 1. Process Data
film_steps, film_avg = get_averaged_data(FILM_LOG_DIR, TARGET_TAGS)
base_steps, base_avg = get_averaged_data(BASELINE_LOG_DIR, TARGET_TAGS, cap_step=BASELINE_CAP)

# 2. Fix Font Issues
# Using 'DejaVu Serif' or 'serif' with a generic fallback to avoid the error
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"], # Fallback chain
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

# 3. Plotting
# FILM
f_smooth = smooth(film_avg, SMOOTHING)
ax.plot(film_steps, f_smooth, label='FiLM + Graph + RWSE', color='#1f77b4', lw=2, zorder=3)
ax.plot(film_steps, film_avg, color='#1f77b4', alpha=0.15, lw=1, zorder=2)

# Baseline
b_smooth = smooth(base_avg, SMOOTHING)
ax.plot(base_steps, b_smooth, label='Baseline', color='#d62728', lw=2, zorder=3)
ax.plot(base_steps, base_avg, color='#d62728', alpha=0.15, lw=1, zorder=2)

# 4. Final Formatting
ax.set_xlabel('Iterations (Steps)')
ax.set_ylabel('Average Episode Length')
ax.set_title('Average Ep Length during Training across variants', fontsize=14, fontweight='bold')
ax.legend(frameon=False, loc='best')
ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_eplen_capped2.pdf', format='pdf', bbox_inches='tight')
plt.show()