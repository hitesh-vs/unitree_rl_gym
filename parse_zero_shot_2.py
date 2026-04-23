import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

# 1. Load the data
data_str = """Variant Model Mean CI Median Max Top10 CI_Top10
all_perturbed_robot_0 film 149.6 10.4 133.0 362 326.9 19.6
all_perturbed_robot_0 baseline 114.9 6.2 110.0 301 225.4 25.7
all_perturbed_robot_1 film 128.1 9.0 110.0 384 311.0 40.5
all_perturbed_robot_1 baseline 116.9 8.0 105.0 328 273.2 28.8
all_perturbed_robot_2 film 152.0 12.6 135.0 706 419.0 112.7
all_perturbed_robot_2 baseline 111.9 6.5 105.0 278 239.8 27.8
all_perturbed_robot_3 film 141.9 11.2 117.0 596 348.6 75.9
all_perturbed_robot_3 baseline 116.1 7.1 109.0 291 270.8 28.9
all_perturbed_robot_4 film 135.2 9.1 125.0 377 308.1 44.3
all_perturbed_robot_4 baseline 106.9 6.1 98.0 267 216.6 30.5
armature_perturbed_robot_0 film 142.8 11.6 124.0 620 376.5 86.6
armature_perturbed_robot_0 baseline 113.7 6.9 107.0 401 256.9 50.6
armature_perturbed_robot_1 film 150.2 11.4 132.0 450 348.8 39.1
armature_perturbed_robot_1 baseline 108.8 6.3 108.0 311 229.2 41.9
armature_perturbed_robot_2 film 140.5 10.2 134.0 447 344.5 53.7
armature_perturbed_robot_2 baseline 116.3 6.7 110.0 304 237.8 29.8
armature_perturbed_robot_3 film 145.6 10.7 135.0 521 351.4 65.6
armature_perturbed_robot_3 baseline 105.2 6.0 97.0 248 218.4 18.6
armature_perturbed_robot_4 film 142.8 12.6 124.0 588 427.0 93.9
armature_perturbed_robot_4 baseline 106.3 6.6 95.0 311 239.5 39.5
damping_perturbed_robot_0 film 146.4 12.7 125.0 707 420.2 116.1
damping_perturbed_robot_0 baseline 114.5 8.0 104.0 472 281.4 60.8
damping_perturbed_robot_1 film 146.1 12.0 127.0 584 381.1 78.8
damping_perturbed_robot_1 baseline 115.7 7.7 104.0 298 262.2 27.5
damping_perturbed_robot_2 film 137.7 9.7 124.0 436 330.8 51.4
damping_perturbed_robot_2 baseline 116.1 8.0 108.0 422 272.5 56.3
damping_perturbed_robot_3 film 137.3 9.1 123.0 477 317.5 66.5
damping_perturbed_robot_3 baseline 107.7 6.6 101.0 303 241.6 37.5
damping_perturbed_robot_4 film 146.0 9.4 133.0 446 318.4 49.4
damping_perturbed_robot_4 baseline 114.5 6.6 110.0 329 241.0 35.4
effort_perturbed_robot_0 film 145.0 10.9 127.0 541 366.6 73.1
effort_perturbed_robot_0 baseline 115.2 6.8 109.0 396 249.6 54.4
effort_perturbed_robot_1 film 148.8 11.1 129.0 501 372.2 63.6
effort_perturbed_robot_1 baseline 108.6 6.8 98.0 297 239.2 32.2
effort_perturbed_robot_2 film 143.7 13.8 119.0 686 452.6 128.7
effort_perturbed_robot_2 baseline 117.2 7.3 109.0 389 266.9 50.6
effort_perturbed_robot_3 film 134.0 9.5 113.0 449 319.6 62.5
effort_perturbed_robot_3 baseline 112.3 7.4 103.0 316 251.2 31.9
effort_perturbed_robot_4 film 138.8 11.0 123.0 471 375.0 70.1
effort_perturbed_robot_4 baseline 111.2 6.6 102.0 311 241.8 33.5
joint_range_perturbed_robot_0 film 142.2 12.1 130.0 766 367.6 116.0
joint_range_perturbed_robot_0 baseline 114.4 6.9 104.0 320 246.2 26.8
joint_range_perturbed_robot_1 film 137.3 9.8 121.0 491 340.5 72.7
joint_range_perturbed_robot_1 baseline 115.3 7.6 101.0 314 270.6 35.6
joint_range_perturbed_robot_2 film 147.3 11.8 125.0 471 391.6 54.4
joint_range_perturbed_robot_2 baseline 116.4 8.2 108.0 450 283.9 57.3
joint_range_perturbed_robot_3 film 139.4 10.1 130.0 388 341.6 49.3
joint_range_perturbed_robot_3 baseline 107.3 6.2 104.0 339 219.4 33.3
joint_range_perturbed_robot_4 film 145.0 12.0 124.0 501 407.9 75.2
joint_range_perturbed_robot_4 baseline 113.5 6.7 108.0 273 238.4 20.6
mass_perturbed_robot_0 film 144.8 11.3 128.0 617 386.4 85.1
mass_perturbed_robot_0 baseline 108.5 6.3 100.0 280 234.6 31.7
mass_perturbed_robot_1 film 142.7 10.0 129.0 387 323.8 41.1
mass_perturbed_robot_1 baseline 108.6 6.2 106.0 302 219.2 34.8
mass_perturbed_robot_2 film 141.0 10.0 132.0 401 327.0 52.2
mass_perturbed_robot_2 baseline 114.8 6.5 110.0 280 233.4 24.6
mass_perturbed_robot_3 film 146.8 10.6 130.0 498 371.1 60.1
mass_perturbed_robot_3 baseline 118.0 8.4 106.0 466 279.0 56.9
mass_perturbed_robot_4 film 146.6 10.9 130.0 495 363.8 66.7
mass_perturbed_robot_4 baseline 113.2 7.6 100.0 346 262.9 41.0
"""

df = pd.read_csv(io.StringIO(data_str), sep=' ')
df['Category'] = df['Variant'].apply(lambda x: x.split('_perturbed')[0])

# Aggregate across seeds
summary = df.groupby(['Category', 'Model']).agg({
    'Top10': 'mean',
    'CI_Top10': 'mean'
}).reset_index()

# 2. Add Flat RL Data
flat_rl_data = {
    'all': (62.8, 10.9), 'armature': (62.0, 10.3), 'damping': (61.0, 10.6),
    'effort': (63.4, 10.6), 'joint_range': (62.7, 10.6), 'mass': (67.0, 11.0)
}

categories = sorted(summary['Category'].unique())
x = np.arange(len(categories))
width = 0.25

# Professional Science Color Palette
c_film = '#2E5A88'  # Deep Steel Blue
c_base = '#D97B42'  # Muted Terracotta
c_flat = '#7F8C8D'  # Slate Gray

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2
})

fig, ax = plt.subplots(figsize=(11, 6))

# Helper to extract values per model
def get_vals(model_name):
    rows = [summary[(summary['Category'] == cat) & (summary['Model'] == model_name)] for cat in categories]
    return [r['Top10'].values[0] for r in rows], [r['CI_Top10'].values[0] for r in rows]

f_mu, f_ci = get_vals('film')
b_mu, b_ci = get_vals('baseline')
rl_mu = [flat_rl_data[c][0] for c in categories]
rl_ci = [flat_rl_data[c][1] for c in categories]

# 3. Draw Bars
ax.bar(x - width, f_mu, width, yerr=f_ci, label='FiLM + Graph + RWSE', 
       color=c_film, edgecolor='black', linewidth=0.5, capsize=3)
ax.bar(x, b_mu, width, yerr=b_ci, label='Baseline', 
       color=c_base, edgecolor='black', linewidth=0.5, capsize=3)
ax.bar(x + width, rl_mu, width, yerr=rl_ci, label='Flat RL Policy', 
       color=c_flat, edgecolor='black', linewidth=0.5, capsize=3)

# 4. Final Formatting
ax.set_ylabel('Avg. Episode Length', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], fontsize=11)

# Grid at every 50
ax.set_yticks(np.arange(0, 501, 50))
ax.yaxis.grid(True, linestyle='--', alpha=0.5)

ax.legend(frameon=False, loc='upper left', fontsize=10)

plt.tight_layout()

# 5. Save and Show
plt.savefig('generalization_performance.pdf', bbox_inches='tight', dpi=300)
plt.show()