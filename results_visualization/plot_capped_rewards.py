# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# # # --- CONFIGURATION ---
# # FILM_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_film_wide/Mar31_18-18-17/events.out.tfevents.1774995525.gpu-5-25.int.turing.wpi.edu.2692739.0'
# # BASELINE_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_baseline_wide/Mar31_18-20-20/events.out.tfevents.1774995648.gpu-5-23.int.turing.wpi.edu.2051735.0'
# # TARGET_TAGS = ['EpLen/variant_robot_variant_4', 'EpLen/variant_robot_variant_6']

# # # UPDATED CAPS
# # FILM_CAP = 300      # Crop blue curve to 300 iters
# # BASELINE_CAP = 420  # Crop red curve to 420 iters
# # SMOOTHING = 0.85

# # def get_averaged_data(log_dir, tags, cap_step=None):
# #     event_acc = EventAccumulator(log_dir)
# #     event_acc.Reload()
    
# #     all_series = []
# #     global_max_step = 0
# #     global_min_step = float('inf')
    
# #     for tag in tags:
# #         events = event_acc.Scalars(tag)
# #         steps = np.array([e.step for e in events])
# #         values = np.array([e.value for e in events])
        
# #         # Apply the cap if specified
# #         if cap_step is not None:
# #             mask = steps <= cap_step
# #             steps = steps[mask]
# #             values = values[mask]
            
# #         if len(steps) == 0: continue
        
# #         all_series.append((steps, values))
# #         global_max_step = max(global_max_step, steps.max())
# #         global_min_step = min(global_min_step, steps.min())

# #     # Create common axis based on the (now capped) max step
# #     common_steps = np.linspace(global_min_step, global_max_step, 1000)
# #     interp_values = [np.interp(common_steps, s, v) for s, v in all_series]
    
# #     return common_steps, np.mean(interp_values, axis=0)

# # def smooth(values, weight):
# #     if len(values) == 0: return np.array([])
# #     last = values[0]
# #     smoothed = []
# #     for point in values:
# #         sampled_avg = last * weight + (1 - weight) * point
# #         smoothed.append(sampled_avg)
# #         last = smoothed[-1]
# #     return np.array(smoothed)

# # # 1. Process Data
# # # Apply FILM_CAP to the blue data
# # film_steps, film_avg = get_averaged_data(FILM_LOG_DIR, TARGET_TAGS, cap_step=FILM_CAP)
# # # Apply BASELINE_CAP to the red data
# # base_steps, base_avg = get_averaged_data(BASELINE_LOG_DIR, TARGET_TAGS, cap_step=BASELINE_CAP)

# # # 2. Fix Font Issues
# # plt.rcParams.update({
# #     "font.family": "serif",
# #     "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"], 
# #     "axes.labelsize": 12,
# #     "xtick.labelsize": 10,
# #     "ytick.labelsize": 10,
# #     "axes.spines.top": False,
# #     "axes.spines.right": False,
# # })

# # fig, ax = plt.subplots(figsize=(7, 4.5))

# # # 3. Plotting
# # # FILM (Blue) - Capped at 300
# # f_smooth = smooth(film_avg, SMOOTHING)
# # ax.plot(film_steps, f_smooth, label='FAGT', color='#1f77b4', lw=2, zorder=3)
# # ax.plot(film_steps, film_avg, color='#1f77b4', alpha=0.15, lw=1, zorder=2)

# # # Baseline (Red) - Capped at 420
# # b_smooth = smooth(base_avg, SMOOTHING)
# # ax.plot(base_steps, b_smooth, label='Baseline (Context Transformer)', color='#d62728', lw=2, zorder=3)
# # ax.plot(base_steps, base_avg, color='#d62728', alpha=0.15, lw=1, zorder=2)

# # # 4. Final Formatting
# # ax.set_xlabel('Iterations (Steps)')
# # ax.set_ylabel('Average Episode Length')
# # ax.set_title('Average Ep Length during Training across variants', fontsize=14, fontweight='bold')
# # ax.legend(frameon=False, loc='lower right') # Moved legend to avoid covering data
# # ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)

# # plt.tight_layout()
# # plt.savefig('comparison_eplen_cropped.pdf', format='pdf', bbox_inches='tight')
# # plt.show()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# # --- CONFIGURATION ---
# FILM_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_film_wide/Mar31_18-18-17/events.out.tfevents.1774995525.gpu-5-25.int.turing.wpi.edu.2692739.0'
# BASELINE_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_baseline_wide/Mar31_18-20-20/events.out.tfevents.1774995648.gpu-5-23.int.turing.wpi.edu.2051735.0'
# TARGET_TAGS = ['EpLen/variant_robot_variant_4', 'EpLen/variant_robot_variant_6']

# BASELINE_CAP = 420  
# SMOOTHING = 0.85

# def get_averaged_data(log_dir, tags, cap_step=None):
#     event_acc = EventAccumulator(log_dir)
#     event_acc.Reload()
#     all_series = []
#     global_max_step = 0
#     global_min_step = float('inf')
    
#     for tag in tags:
#         events = event_acc.Scalars(tag)
#         steps = np.array([e.step for e in events])
#         values = np.array([e.value for e in events])
#         if cap_step is not None:
#             mask = steps <= cap_step
#             steps = steps[mask]; values = values[mask]
#         if len(steps) == 0: continue
#         all_series.append((steps, values))
#         global_max_step = max(global_max_step, steps.max())
#         global_min_step = min(global_min_step, steps.min())

#     common_steps = np.linspace(global_min_step, global_max_step, 1000)
#     interp_values = [np.interp(common_steps, s, v) for s, v in all_series]
#     return common_steps, np.mean(interp_values, axis=0)

# def smooth(values, weight):
#     if len(values) == 0: return np.array([])
#     last = values[0]
#     smoothed = []
#     for point in values:
#         sampled_avg = last * weight + (1 - weight) * point
#         smoothed.append(sampled_avg); last = smoothed[-1]
#     return np.array(smoothed)

# # 1. Process Baseline (Red) to find its max value
# base_steps, base_avg = get_averaged_data(BASELINE_LOG_DIR, TARGET_TAGS, cap_step=BASELINE_CAP)
# b_smooth = smooth(base_avg, SMOOTHING)
# max_red_val = np.max(b_smooth)

# # 2. Process FILM (Blue) and find where it hits that value
# # We fetch the full blue data first to find the intersection point
# film_steps_full, film_avg_full = get_averaged_data(FILM_LOG_DIR, TARGET_TAGS)
# f_smooth_full = smooth(film_avg_full, SMOOTHING)

# # Find first index where blue exceeds or equals the red max
# # If it never reaches it, we keep the full graph
# cross_indices = np.where(f_smooth_full >= max_red_val)[0]
# if len(cross_indices) > 0:
#     first_cross_idx = cross_indices[0]
#     dynamic_film_cap = film_steps_full[first_cross_idx]
# else:
#     dynamic_film_cap = film_steps_full[-1]

# # 3. Re-process/Crop Blue data at that specific iteration
# film_steps, film_avg = get_averaged_data(FILM_LOG_DIR, TARGET_TAGS, cap_step=dynamic_film_cap)
# f_smooth = smooth(film_avg, SMOOTHING)

# print(f"Max Baseline Ep Length: {max_red_val:.2f}")
# print(f"Blue curve reaches this at iteration: {dynamic_film_cap:.0f}")

# # 4. Plotting
# plt.rcParams.update({
#     "font.family": "serif",
#     "axes.spines.top": False,
#     "axes.spines.right": False,
# })

# fig, ax = plt.subplots(figsize=(7, 4.5))

# # Blue (FAGT) - Cropped at the Y-value match
# ax.plot(film_steps, f_smooth, label='FAGT', color='#1f77b4', lw=2, zorder=3)
# ax.plot(film_steps, film_avg, color='#1f77b4', alpha=0.15, lw=1, zorder=2)

# # Red (Baseline)
# ax.plot(base_steps, b_smooth, label='Baseline (Context Transformer)', color='#d62728', lw=2, zorder=3)
# ax.plot(base_steps, base_avg, color='#d62728', alpha=0.15, lw=1, zorder=2)

# # Horizontal line showing the target value
# ax.axhline(y=max_red_val, color='gray', linestyle='--', alpha=0.4, label=f'Peak Y: {max_red_val:.1f}')

# ax.set_xlabel('Iterations (Steps)')
# ax.set_ylabel('Average Episode Length')
# ax.set_title('Training Comparison (Blue Capped at Red Peak Y)', fontsize=12, fontweight='bold')
# ax.legend(frameon=False, loc='lower right')
# ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# plt.tight_layout()
# plt.savefig('comparison_eplen_y_capped.pdf', format='pdf', bbox_inches='tight')
# plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
FILM_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_film_wide/Mar31_18-18-17/events.out.tfevents.1774995525.gpu-5-25.int.turing.wpi.edu.2692739.0'
BASELINE_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_baseline_wide/Mar31_18-20-20/events.out.tfevents.1774995648.gpu-5-23.int.turing.wpi.edu.2051735.0'
TARGET_TAGS = ['EpLen/variant_robot_variant_4', 'EpLen/variant_robot_variant_6']

FILM_CAP = 300      # Strictly capped at 300
BASELINE_CAP = 420  # Strictly capped at 420
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
        if cap_step is not None:
            mask = steps <= cap_step
            steps = steps[mask]; values = values[mask]
        if len(steps) == 0: continue
        all_series.append((steps, values))
        global_max_step = max(global_max_step, steps.max())
        global_min_step = min(global_min_step, steps.min())

    common_steps = np.linspace(global_min_step, global_max_step, 1000)
    interp_values = [np.interp(common_steps, s, v) for s, v in all_series]
    return common_steps, np.mean(interp_values, axis=0)

def smooth(values, weight):
    if len(values) == 0: return np.array([])
    last = values[0]
    smoothed = []
    for point in values:
        sampled_avg = last * weight + (1 - weight) * point
        smoothed.append(sampled_avg); last = smoothed[-1]
    return np.array(smoothed)

# 1. Process Data with fixed caps
film_steps, film_avg = get_averaged_data(FILM_LOG_DIR, TARGET_TAGS, cap_step=FILM_CAP)
base_steps, base_avg = get_averaged_data(BASELINE_LOG_DIR, TARGET_TAGS, cap_step=BASELINE_CAP)

# 2. Apply Smoothing
f_smooth = smooth(film_avg, SMOOTHING)
b_smooth = smooth(base_avg, SMOOTHING)

# 3. Calculate the max Y of Red for the dotted line
max_red_val = np.max(b_smooth)

# 4. Plotting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"], 
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

# Blue (FAGT) - Capped at 300
ax.plot(film_steps, f_smooth, label='FAGT', color='#1f77b4', lw=2, zorder=3)
ax.plot(film_steps, film_avg, color='#1f77b4', alpha=0.15, lw=1, zorder=2)

# Red (Baseline) - Capped at 420
ax.plot(base_steps, b_smooth, label='Baseline (Context Transformer)', color='#d62728', lw=2, zorder=3)
ax.plot(base_steps, base_avg, color='#d62728', alpha=0.15, lw=1, zorder=2)

# ADD DOTTED LINE: From X=0 to the end of the graph (420) at Red's Max Y
ax.axhline(y=max_red_val, color='black', linestyle='--', alpha=0.6, lw=1.2, 
           label=f'Baseline Peak ({max_red_val:.1f})')

# 5. Final Formatting
ax.set_xlabel('Training Iterations')
ax.set_ylabel('Average Simulation Steps')
ax.set_title('Average Simulation Steps during Training across variants', fontsize=14, fontweight='bold')
ax.legend(frameon=False, loc='lower right')
ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_eplen_with_threshold2.pdf', format='pdf', bbox_inches='tight')
print(f"Max Simulation Steps reached by Baseline: {max_red_val:.2f}")
plt.show()