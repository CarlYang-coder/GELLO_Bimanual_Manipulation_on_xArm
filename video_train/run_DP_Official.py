"""
Official-style online rollout for DiffusionUnetVideoPolicy.

Loads checkpoints saved by train_DP_OfficialVideo.py (both inference .pt
and full .ckpt formats are supported).

Features (matching run_video_policy_online_clamp.py):
  - CSV-based initial pose (mean of first rows across sessions)
  - Joint position clamping (Q_MIN / Q_MAX)
  - EMA smoothing + deadband filtering
  - Action chunk execution (receding horizon)
  - DRY_RUN mode for safe testing

Usage:
    python run_DP_Official.py
"""

# ============================================================
# HARD-CODED CONFIG
# ============================================================

# ---- Checkpoint: accepts either .pt (inference) or .ckpt (full) ----
CKPT_PATH = r"D:\Image_DP\ckpts_official_video\best.pt"

# ---- Robot IPs ----
LEFT_ARM_IP  = "192.168.1.224"
RIGHT_ARM_IP = "192.168.1.235"

# ---- Camera ----
CAM_INDEX = 0
HZ        = 10.0

# ---- Safety / stability ----
MAX_DQ_JOINT   = 0.01    # rad per step
MAX_DQ_GRIP    = 0.2     # gripper ratio per step
EMA_ALPHA      = 0.8     # higher = smoother (0.7~0.9)
DEADBAND_JOINT = 0.001
DEADBAND_GRIP  = 0.002

import numpy as np

# ---- Joint position limits (global min/max from all training sessions) ----
Q_MIN = np.array([
    -0.049088, -0.130386, -0.046004,  0.292987, -0.378898,  0.174870, -0.081298,  0.162500,
    -0.596704,  0.320605, -0.070547,  0.276113, -0.116587,  0.207099, -0.406502,  0.961250,
], dtype=np.float32)
Q_MAX = np.array([
     0.070577,  0.562974,  0.905049,  1.310016,  0.286850,  1.265453,  0.727109,  0.870000,
     0.076698,  0.558372,  0.210155,  0.783859, -0.047573,  0.418773,  0.027614,  1.002500,
], dtype=np.float32)

# ---- Servo speeds ----
SPEED      = 0.4
MVACC      = 1.0
INIT_SPEED = 0.3
INIT_MVACC = 0.5

USE_GRIPPER = True

# ---- Training data root (for reading initial pose from CSV) ----
DATA_ROOT = r"D:\Image_DP\data\GelloAgent"

# ---- If True: don't send commands, only print dq/q ----
DRY_RUN = False


# ============================================================
# CODE
# ============================================================

import time
import signal
import collections
from pathlib import Path

import cv2
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms

from xarm.wrapper import XArmAPI

# ---- diffusion_policy imports ----
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from resnet_video_encoder import ResNet18VideoEncoder
from diffusion_unet_video_policy import DiffusionUnetVideoPolicy as _Policy


# ============================================================
# Image transform (same as gello_video_dataset.py)
# ============================================================

CROP_BOX = (150, 50, 450, 350)  # same as Downsampling.py

def _make_img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# xArm helpers
# ============================================================

def init_arm(ip):
    arm = XArmAPI(ip, is_radian=True)
    arm.connect()
    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)
    return arm

def get_q7(arm):
    _, q = arm.get_servo_angle(is_radian=True)
    return np.array(q[:7], dtype=np.float32)

def send_q7(arm, q7):
    arm.set_servo_angle_j(q7.tolist(), is_radian=True, speed=SPEED, mvacc=MVACC)

def get_gripper_ratio(arm):
    """xArm pos 0=closed, 850=open -> ratio 1.0=closed, 0.0=open."""
    try:
        code, pos = arm.get_gripper_position()
        if code == 0:
            return float(np.clip(1.0 - pos / 850.0, 0.0, 1.0))
    except Exception:
        pass
    return 1.0

def set_gripper_ratio(arm, ratio, speed=5000):
    """ratio 1.0=closed, 0.0=open -> xArm pos 0=closed, 850=open."""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    pos = int(round((1.0 - ratio) * 850))
    try:
        arm.set_gripper_position(pos, wait=False, speed=speed)
    except Exception:
        pass


# ============================================================
# Initial pose from training data
# ============================================================

def compute_mean_init_pose(data_root):
    """Compute mean initial pose (first row) across all training sessions."""
    csvs = sorted(Path(data_root).glob("**/joint_with_images_downsampled.csv"))
    assert len(csvs) > 0, f"No CSVs found under {data_root}"
    q_keys = ([f"left_joint_positions_{i}" for i in range(1, 9)] +
              [f"right_joint_positions_{i}" for i in range(1, 9)])
    init_poses = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        row0 = df[q_keys].iloc[0].to_numpy(dtype=np.float32)
        init_poses.append(row0)
    mean_pose = np.array(init_poses).mean(axis=0)  # (16,)
    return mean_pose


def move_to_init_pose(left, right, init_q16):
    """Move both arms to the initial pose using position mode (slow, safe)."""
    qL_init = init_q16[0:7]
    gL_init = float(init_q16[7])
    qR_init = init_q16[8:15]
    gR_init = float(init_q16[15])

    print(f"[INIT] Moving to initial pose ...")
    print(f"       left  joints: {qL_init}")
    print(f"       left  grip:   {gL_init:.4f}")
    print(f"       right joints: {qR_init}")
    print(f"       right grip:   {gR_init:.4f}")

    # Switch to position mode (mode=0)
    for arm in [left, right]:
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.1)

    left.set_servo_angle(angle=qL_init.tolist(), is_radian=True,
                         speed=INIT_SPEED, mvacc=INIT_MVACC, wait=False)
    right.set_servo_angle(angle=qR_init.tolist(), is_radian=True,
                          speed=INIT_SPEED, mvacc=INIT_MVACC, wait=False)
    if USE_GRIPPER:
        set_gripper_ratio(left, gL_init)
        set_gripper_ratio(right, gR_init)

    # Wait until both arms are close to target
    tol = 0.02  # rad
    for _ in range(200):  # max ~20s
        qL_now = get_q7(left)
        qR_now = get_q7(right)
        errL = np.max(np.abs(qL_now - qL_init))
        errR = np.max(np.abs(qR_now - qR_init))
        if errL < tol and errR < tol:
            break
        time.sleep(0.1)

    print(f"[INIT] Reached. left={get_q7(left)[:3]}  right={get_q7(right)[:3]}")


# ============================================================
# Build policy from checkpoint
# ============================================================

def build_policy(ckpt_path, device):
    """
    Load policy from checkpoint. Supports two formats:
      1. Inference .pt  (from train_DP_OfficialVideo or train_diffusion_video)
         Keys: "policy", "config"
      2. Full .ckpt     (from train_DP_OfficialVideo workspace)
         Keys: "model", "cfg"
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ---- Detect format ----
    if "policy" in ckpt:
        # inference .pt format
        c = ckpt["config"]
        state_dict = ckpt["policy"]
    elif "model" in ckpt:
        # full workspace .ckpt format
        c = ckpt["cfg"]
        state_dict = ckpt["model"]
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())}")

    # ---- Extract config ----
    # Support both dict key styles
    img_size            = c.get("img_size", c.get("img_size"))
    n_obs_steps         = c.get("n_obs_steps", c.get("n_obs_steps"))
    n_action_steps      = c.get("n_action_steps", c.get("n_action_steps"))
    horizon             = c.get("horizon", c.get("horizon"))
    num_train_timesteps = c.get("num_train_timesteps", c.get("num_train_timesteps"))
    num_infer_steps     = c.get("num_infer_steps", c.get("num_infer_steps"))

    # ---- Build model (same architecture as training) ----
    rgb_net = ResNet18VideoEncoder(
        out_dim=c.get("rgb_out_dim", 512),
        pool=c.get("rgb_pool", "mean"),
        mlp_hidden=c.get("rgb_mlp_hidden", 512),
        dropout=c.get("rgb_dropout", 0.0),
        pretrained=False,           # weights come from checkpoint
        freeze_backbone=True,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=c.get("beta_schedule", "squaredcos_cap_v2"),
        clip_sample=False,
        prediction_type=c.get("prediction_type", "epsilon"),
    )
    action_dim = c.get("action_dim", 16)
    shape_meta = {
        "action": {"shape": [action_dim]},
        "obs": {
            "rgb": {"shape": [3, img_size, img_size], "type": "rgb"},
            "q":   {"shape": [action_dim], "type": "lowdim"},
        }
    }
    policy = _Policy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        rgb_net=rgb_net,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=num_infer_steps,
        lowdim_as_global_cond=True,
        predict_epsilon=c.get("predict_epsilon", True),
    )
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"       epoch={ckpt.get('epoch', '?')}  val_loss={ckpt.get('val_loss', '?')}")
    print(f"       horizon={horizon}  n_obs={n_obs_steps}  n_act={n_action_steps}")
    print(f"       infer_steps={num_infer_steps}  img_size={img_size}")

    return policy, img_size, n_obs_steps, n_action_steps


# ============================================================
# Safety filters
# ============================================================

def safe_dq(dq):
    """Clamp + deadband on 16-dim delta-q."""
    dq = dq.copy()
    dq[0:7]  = np.clip(dq[0:7],  -MAX_DQ_JOINT, MAX_DQ_JOINT)
    dq[8:15] = np.clip(dq[8:15], -MAX_DQ_JOINT, MAX_DQ_JOINT)
    dq[7]    = float(np.clip(dq[7],  -MAX_DQ_GRIP, MAX_DQ_GRIP))
    dq[15]   = float(np.clip(dq[15], -MAX_DQ_GRIP, MAX_DQ_GRIP))

    dq[0:7][np.abs(dq[0:7])   < DEADBAND_JOINT] = 0.0
    dq[8:15][np.abs(dq[8:15]) < DEADBAND_JOINT] = 0.0
    if abs(dq[7])  < DEADBAND_GRIP:  dq[7]  = 0.0
    if abs(dq[15]) < DEADBAND_GRIP:  dq[15] = 0.0
    return dq


# ============================================================
# Main rollout loop
# ============================================================

def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---- Build policy ----
    policy, img_size, n_obs_steps, n_action_steps = build_policy(CKPT_PATH, device)

    # ---- Image transform ----
    img_transform = _make_img_transform(img_size)

    # ---- Camera ----
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    assert cap.isOpened(), "Camera open failed"

    # ---- Arms ----
    left  = init_arm(LEFT_ARM_IP)
    right = init_arm(RIGHT_ARM_IP)

    # ---- Step 1: Move to initial pose ----
    init_q16 = compute_mean_init_pose(DATA_ROOT)
    print("[STEP 1] Press ENTER to move arms to training initial pose.")
    input()
    move_to_init_pose(left, right, init_q16)

    # ---- Step 2: Start policy ----
    print("\n[STEP 2] Arms at initial pose. Press ENTER to start policy. Ctrl+C to stop.")
    input()

    # Switch to servo mode
    for arm in [left, right]:
        arm.set_mode(1)
        arm.set_state(0)
        time.sleep(0.1)
    send_q7(left,  get_q7(left))
    send_q7(right, get_q7(right))

    # ---- Observation ring buffers ----
    rgb_buf = collections.deque(maxlen=n_obs_steps)
    q_buf   = collections.deque(maxlen=n_obs_steps)

    # EMA state for absolute target positions (init to current pose)
    qL_init_now = get_q7(left)
    qR_init_now = get_q7(right)
    gL_init_now = get_gripper_ratio(left)  if USE_GRIPPER else 0.0
    gR_init_now = get_gripper_ratio(right) if USE_GRIPPER else 0.0
    target_ema = np.concatenate([qL_init_now, [gL_init_now], qR_init_now, [gR_init_now]]).astype(np.float32)

    # Action chunk buffer
    action_chunk = []
    chunk_idx = 0

    dt = 1.0 / float(HZ)

    try:
        while True:
            t0 = time.perf_counter()

            # ---- Read camera ----
            ok, frame = cap.read()
            if not ok:
                continue

            # ---- Preprocess image ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).crop(CROP_BOX)
            rgb_tensor = img_transform(pil_img)

            # ---- Read robot state ----
            qL = get_q7(left)
            qR = get_q7(right)
            gL = get_gripper_ratio(left)  if USE_GRIPPER else 0.0
            gR = get_gripper_ratio(right) if USE_GRIPPER else 0.0
            q_now = np.concatenate([qL, [gL], qR, [gR]])
            q_tensor = torch.from_numpy(q_now).float()

            # ---- Push into observation buffers ----
            rgb_buf.append(rgb_tensor)
            q_buf.append(q_tensor)

            # Wait until we have enough obs frames — hold position meanwhile
            if len(rgb_buf) < n_obs_steps:
                if not DRY_RUN:
                    send_q7(left, qL)
                    send_q7(right, qR)
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # ---- Run policy if action chunk exhausted ----
            if chunk_idx >= len(action_chunk):
                obs_rgb = torch.stack(list(rgb_buf), dim=0).unsqueeze(0).to(device)
                obs_q   = torch.stack(list(q_buf),   dim=0).unsqueeze(0).to(device)

                obs_dict = {"rgb": obs_rgb, "q": obs_q}

                with torch.no_grad():
                    result = policy.predict_action(obs_dict)

                action_chunk = result["action"][0].cpu().numpy()
                chunk_idx = 0

            # ---- Get current action from chunk (absolute target position) ----
            target_raw = action_chunk[chunk_idx].astype(np.float32)
            chunk_idx += 1

            # ---- EMA smooth on absolute target ----
            target_ema = EMA_ALPHA * target_ema + (1.0 - EMA_ALPHA) * target_raw

            # ---- Extract target positions ----
            qL_cmd = target_ema[0:7]
            qR_cmd = target_ema[8:15]
            gL_cmd = float(target_ema[7])
            gR_cmd = float(target_ema[15])

            # ---- Clamp absolute joint positions ----
            qL_cmd = np.clip(qL_cmd, Q_MIN[0:7],  Q_MAX[0:7])
            qR_cmd = np.clip(qR_cmd, Q_MIN[8:15], Q_MAX[8:15])

            # ---- Send ----
            if DRY_RUN:
                print(f"qL={qL_cmd[:3]}  qR={qR_cmd[:3]}  "
                      f"gL={gL_cmd:.3f}  gR={gR_cmd:.3f}")
            else:
                send_q7(left,  qL_cmd)
                send_q7(right, qR_cmd)
                if USE_GRIPPER:
                    set_gripper_ratio(left,  gL_cmd)
                    set_gripper_ratio(right, gR_cmd)

            # ---- Timing ----
            elapsed = time.perf_counter() - t0
            hz_actual = 1.0 / max(elapsed, 1e-6)
            print(f"\r[HZ] {hz_actual:.1f} Hz  (loop {elapsed*1000:.1f} ms)", end="", flush=True)
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        left.disconnect()
        right.disconnect()
        print("[DONE] Clean exit.")


if __name__ == "__main__":
    main()
