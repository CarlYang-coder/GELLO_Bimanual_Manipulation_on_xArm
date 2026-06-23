import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path(r"D:\Image_DP\data\GelloAgent")
csvs = sorted(DATA_ROOT.glob("**/joint_with_images_downsampled.csv"))

q_keys = [f"left_joint_positions_{i}" for i in range(1, 9)] + [f"right_joint_positions_{i}" for i in range(1, 9)]

all_q = []
init_poses = []
for csv_path in csvs:
    df = pd.read_csv(csv_path)
    q = df[q_keys].to_numpy(dtype=np.float32)
    all_q.append(q)
    init_poses.append(q[0])

all_q = np.concatenate(all_q, axis=0)
init_poses = np.array(init_poses)

print(f"Total frames: {all_q.shape[0]}, Sessions: {len(init_poses)}")
print()

# Focus on L2 (index 1)
l2 = all_q[:, 1]
print("=== L2 (left_joint_positions_2) ===")
print(f"  min={l2.min():.4f}  max={l2.max():.4f}  mean={l2.mean():.4f}  std={l2.std():.4f}")
for p in [1, 5, 10, 15, 20, 25, 30, 50, 75, 80, 85, 90, 95, 99]:
    print(f"  P{p:02d} = {np.percentile(l2, p):.4f}")

print()
print("=== L2 initial poses (first frame of each session) ===")
l2_init = init_poses[:, 1]
print(f"  min={l2_init.min():.4f}  max={l2_init.max():.4f}  mean={l2_init.mean():.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:02d} = {np.percentile(l2_init, p):.4f}")

print()
print("=== All 16 joints: P5 / P10 / P25 / P50 / P75 / P90 / P95 ===")
names = [f"L{i}" for i in range(1,9)] + [f"R{i}" for i in range(1,9)]
for i, name in enumerate(names):
    vals = all_q[:, i]
    ps = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
    print(f"  {name}: P5={ps[0]:.4f}  P10={ps[1]:.4f}  P25={ps[2]:.4f}  P50={ps[3]:.4f}  P75={ps[4]:.4f}  P90={ps[5]:.4f}  P95={ps[6]:.4f}")

print()
print("=== Current Q_MIN vs data range ===")
Q_MIN_current = np.array([-0.0245, 0.3000, 0.0850, 0.4740, -0.2961, 0.3482, 0.0215, 0.1713,
                           -0.4741, 0.3666, -0.0368, 0.3789, -0.1120, 0.2301, -0.2629, 0.9812])
for i, name in enumerate(names):
    vals = all_q[:, i]
    p5 = np.percentile(vals, 5)
    print(f"  {name}: Q_MIN={Q_MIN_current[i]:.4f}  P5={p5:.4f}  P1={np.percentile(vals, 1):.4f}  min={vals.min():.4f}")
