import pickle
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────
with open("trajectory.pkl", "rb") as f:
    trajectory = pickle.load(f)

simulation_dt      = 0.002
control_decimation = 10
effective_dt       = simulation_dt * control_decimation

all_qpos = np.array([frame['qpos'][7:] for frame in trajectory])  # (T, 12)

# ── Discard settling period ───────────────────────────────────────────────
settle_frames = int(3.0 / simulation_dt)
qpos_settled  = all_qpos[settle_frames:]

# ── Extract one gait cycle ────────────────────────────────────────────────
frames_per_cycle = int(0.8 / simulation_dt)  # 400 frames
mid        = len(qpos_settled) // 2
gait_cycle = qpos_settled[mid : mid + frames_per_cycle]

np.save("gait_cycle_clean.npy", gait_cycle)
print(f"Saved gait cycle: {gait_cycle.shape}")  # expect (400, 12)

# ── Basic sanity check without matplotlib ────────────────────────────────
print(f"Joint angle ranges across cycle:")
for i in range(12):
    print(f"  joint {i:2d}: min={gait_cycle[:,i].min():.3f}  max={gait_cycle[:,i].max():.3f}  range={gait_cycle[:,i].ptp():.3f}")