import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
# Replace these with the full paths to your log FOLDERS
FILM_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_film_finetune/Apr13_16-26-59/'
BASELINE_LOG_DIR = '/home/sviswasam/dr/unitree_rl_gym/output_base_finetune/Apr13_16-48-25'
OUTPUT_NAME = 'eplen_averaged_comparison.pdf'
SMOOTHING = 0.85

def get_run_average(log_dir):
    """
    Finds the event file, averages ALL EpLen tags (variants and robot types), 
    and returns a single set of steps and averaged values.
    """
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return None, None
        
    files = glob.glob(os.path.join(log_dir, "*.tfevents*"))
    if not files:
        files = glob.glob(os.path.join(log_dir, "**/*.tfevents*"), recursive=True)
    
    if not files:
        print(f"No tfevents in {log_dir}")
        return None, None

    # Always use the latest event file in the folder
    ea = EventAccumulator(max(files, key=os.path.getmtime), size_guidance={'scalars': 0})
    ea.Reload()
    
    # Grab every tag that has 'EpLen' in it
    tags = ea.Tags()['scalars']
    eplen_tags = [t for t in tags if 'EpLen' in t]
    
    if not eplen_tags:
        print(f"No EpLen tags found in {log_dir}")
        return None, None

    print(f"Averaging {len(eplen_tags)} variants for: {os.path.basename(log_dir)}")
    
    all_values = []
    
    # We find the longest tag to use as our base step-axis
    max_steps_tag = max(eplen_tags, key=lambda t: len(ea.Scalars(t)))
    base_events = ea.Scalars(max_steps_tag)
    common_steps = np.array([e.step for e in base_events])
    
    for tag in eplen_tags:
        evs = ea.Scalars(tag)
        steps = np.array([e.step for e in evs])
        vals = np.array([e.value for e in evs])
        
        # Interpolate this variant's data to the common_steps axis
        interp_v = np.interp(common_steps, steps, vals)
        all_values.append(interp_v)
        
    final_avg = np.mean(all_values, axis=0)
    return common_steps, final_avg

def smooth(values, weight):
    if values is None or len(values) == 0: return values
    last = values[0]
    smoothed = []
    for point in values:
        sampled_avg = last * weight + (1 - weight) * point
        smoothed.append(sampled_avg)
        last = smoothed[-1]
    return np.array(smoothed)

def plot():
    # Style Setup
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Process both runs
    f_x, f_y = get_run_average(FILM_LOG_DIR)
    b_x, b_y = get_run_average(BASELINE_LOG_DIR)

    # Plot FiLM
    if f_y is not None:
        ax.plot(f_x, smooth(f_y, SMOOTHING), label='FiLM + Graph (Averaged)', color='#1f77b4', lw=2.5, zorder=3)
        ax.plot(f_x, f_y, color='#1f77b4', alpha=0.1, lw=1)
        
    # Plot Baseline
    if b_y is not None:
        ax.plot(b_x, smooth(b_y, SMOOTHING), label='Baseline (Averaged)', color='#d62728', lw=2.5, zorder=3)
        ax.plot(b_x, b_y, color='#d62728', alpha=0.1, lw=1)

    if f_y is None and b_y is None:
        print("\n!!! ERROR: No data found. Please check your folder paths.")
        return

    ax.set_title('Average Episode Length across all Robots/Variants', fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean EpLen')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_NAME, bbox_inches='tight')
    print(f"\nSuccess! Comparison saved to {OUTPUT_NAME}")

if __name__ == "__main__":
    plot()