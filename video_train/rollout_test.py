# -*- coding: utf-8 -*-
r"""
Rollout test: left arm replays ground-truth from CSV, right arm runs policy.

- Left arm: absolute joint positions from CSV (ground truth), sent directly
- Right arm: policy predict_action (delta-q), applied to current state
- Press ENTER to start, Ctrl+C to stop

Act: 16D = [left 7 joints + left gripper, right 7 joints + right gripper]
"""

# ============================================================
# HARD-CODED CONFIG
# ============================================================

CKPT_PATH = r"D:\Image_DP\ckpts_video_downsampled\best.pt"
CSV_PATH  = r"D:\Image_DP\data\GelloAgent\0209_190553\joint_with_images.csv"

LEFT_ARM_IP  = "192.168.1.224"
RIGHT_ARM_IP = "192.168.1.235"

CAM_INDEX = 0
HZ        = 15.0

# ---- Safety / stability (for right arm policy only) ----
MAX_DQ_JOINT = 0.01
MAX_DQ_GRIP  = 0.02
EMA_ALPHA    = 0.8
DEADBAND_JOINT = 0.001
DEADBAND_GRIP  = 0.002

SPEED = 0.4
MVACC = 1.0

USE_GRIPPER = True
DRY_RUN = False

# ============================================================
# CODE
# ============================================================

import time
import signal
import collections

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import transforms

from xarm.wrapper import XArmAPI

# ---- diffusion_policy imports (same as train_diffusion_video.py) ----
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from resnet_video_encoder import ResNet18VideoEncoder
from diffusion_unet_video_policy import DiffusionUnetVideoPolicy as _Policy


# ---------------------------
# Image transform (same as gello_video_dataset.py _default_img_transform)
# ---------------------------

def _make_img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------
# xArm helpers (from run_policy_online.py)
# ---------------------------

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
    try:
        code, pos = arm.get_gripper_position()
        if code == 0:
            return float(np.clip(pos / 850.0, 0.0, 1.0))
    except Exception:
        pass
    return 0.0

def set_gripper_ratio(arm, ratio, speed=5000):
    ratio = float(np.clip(ratio, 0.0, 1.0))
    pos = int(round(ratio * 850))
    try:
        arm.set_gripper_position(pos, wait=False, speed=speed)
    except Exception:
        pass


# ---------------------------
# Build policy from checkpoint (mirrors train_diffusion_video.py)
# ---------------------------

def build_policy(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    IMG_SIZE           = cfg["img_size"]
    N_OBS_STEPS        = cfg["n_obs_steps"]
    N_ACTION_STEPS     = cfg["n_action_steps"]
    HORIZON            = cfg["horizon"]
    NUM_TRAIN_TIMESTEPS = cfg["num_train_timesteps"]
    NUM_INFER_STEPS    = cfg["num_infer_steps"]

    rgb_net = ResNet18VideoEncoder(
        out_dim=512, pool="mean", mlp_hidden=512,
        dropout=0.0, pretrained=False, freeze_backbone=True,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False, prediction_type="epsilon",
    )

    shape_meta = {
        "action": {"shape": [16]},
        "obs": {
            "rgb": {"shape": [3, IMG_SIZE, IMG_SIZE], "type": "rgb"},
            "q":   {"shape": [16], "type": "lowdim"},
        }
    }

    policy = _Policy(
        shape_meta=shape_meta, noise_scheduler=noise_scheduler,
        rgb_net=rgb_net, horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS, n_obs_steps=N_OBS_STEPS,
        num_inference_steps=NUM_INFER_STEPS,
        lowdim_as_global_cond=True, predict_epsilon=True,
    )

    policy.load_state_dict(ckpt["policy"])
    policy = policy.to(device)
    policy.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"       epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")
    print(f"       horizon={HORIZON}  n_obs={N_OBS_STEPS}  n_act={N_ACTION_STEPS}")
    print(f"       infer_steps={NUM_INFER_STEPS}  img_size={IMG_SIZE}")

    return policy, IMG_SIZE, N_OBS_STEPS, N_ACTION_STEPS


# ---------------------------
# Load ground-truth CSV for left arm
# ---------------------------

def load_gt_csv(csv_path):
    """Load CSV and return left arm absolute joint positions (N, 8) and right (N, 8)."""
    df = pd.read_csv(csv_path)
    left_keys  = [f"left_joint_positions_{i}" for i in range(1, 9)]
    right_keys = [f"right_joint_positions_{i}" for i in range(1, 9)]
    q_left  = df[left_keys].to_numpy(dtype=np.float32)   # (N, 8) = 7 joints + 1 gripper
    q_right = df[right_keys].to_numpy(dtype=np.float32)  # (N, 8) = 7 joints + 1 gripper
    print(f"[INFO] Loaded GT CSV: {csv_path}  ({len(df)} rows)")
    return q_left, q_right


# ---------------------------
# Safety filter for right arm delta-q
# ---------------------------

def safe_dq(dq):
    """Clamp, deadband on 16-dim delta-q."""
    dq = dq.copy()
    dq[0:7]  = np.clip(dq[0:7],  -MAX_DQ_JOINT, MAX_DQ_JOINT)
    dq[8:15] = np.clip(dq[8:15], -MAX_DQ_JOINT, MAX_DQ_JOINT)
    dq[7]    = float(np.clip(dq[7],  -MAX_DQ_GRIP, MAX_DQ_GRIP))
    dq[15]   = float(np.clip(dq[15], -MAX_DQ_GRIP, MAX_DQ_GRIP))

    dq[0:7][np.abs(dq[0:7])   < DEADBAND_JOINT] = 0.0
    dq[8:15][np.abs(dq[8:15]) < DEADBAND_JOINT] = 0.0
    if abs(dq[7])  < DEADBAND_GRIP: dq[7]  = 0.0
    if abs(dq[15]) < DEADBAND_GRIP: dq[15] = 0.0
    return dq


# ---------------------------
# Main loop
# ---------------------------

def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---- Build policy for right arm ----
    policy, img_size, n_obs_steps, n_action_steps = build_policy(CKPT_PATH, device)

    # ---- Load ground-truth for left arm ----
    gt_left, gt_right = load_gt_csv(CSV_PATH)
    gt_total = len(gt_left)

    # ---- Image transform (same as gello_video_dataset.py) ----
    img_transform = _make_img_transform(img_size)

    # ---- Camera ----
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    assert cap.isOpened(), "Camera open failed"

    # ---- Arms ----
    left  = init_arm(LEFT_ARM_IP)
    right = init_arm(RIGHT_ARM_IP)

    # ---- Observation ring buffers (for right arm policy) ----
    rgb_buf = collections.deque(maxlen=n_obs_steps)
    q_buf   = collections.deque(maxlen=n_obs_steps)

    # EMA state (for right arm)
    dq_ema = np.zeros(16, dtype=np.float32)

    # Action chunk buffer (for right arm)
    action_chunk = []
    chunk_idx = 0

    # CSV row counter (for left arm ground truth)
    gt_idx = 0

    print(f"[READY] Left arm: GT replay ({gt_total} steps)  |  Right arm: policy")
    print(f"        Press ENTER to start. Ctrl+C to stop.")
    input()

    dt = 1.0 / float(HZ)

    try:
        while True:
            t0 = time.perf_counter()

            # ---- Check if GT exhausted ----
            if gt_idx >= gt_total:
                print(f"\n[INFO] Ground-truth replay finished ({gt_total} steps). Stopping.")
                break

            # ---- Read camera ----
            ok, frame = cap.read()
            if not ok:
                continue

            # ---- Preprocess image (same as gello_video_dataset.py) ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            rgb_tensor = img_transform(pil_img)  # (3, H, W)

            # ---- Read robot state ----
            qL = get_q7(left)
            qR = get_q7(right)
            gL = get_gripper_ratio(left)  if USE_GRIPPER else 0.0
            gR = get_gripper_ratio(right) if USE_GRIPPER else 0.0
            q_now = np.concatenate([qL, [gL], qR, [gR]])  # (16,)
            q_tensor = torch.from_numpy(q_now).float()

            # ---- Push into observation buffers (for policy) ----
            rgb_buf.append(rgb_tensor)
            q_buf.append(q_tensor)

            # ============================================
            # LEFT ARM: ground-truth absolute positions
            # ============================================
            gt_row = gt_left[gt_idx]          # (8,) = 7 joints + 1 gripper
            gt_left_q7 = gt_row[:7]           # 7 joint positions
            gt_left_grip = float(gt_row[7])   # gripper ratio

            if DRY_RUN:
                print(f"\n[GT L] idx={gt_idx}  q7={gt_left_q7}  grip={gt_left_grip:.4f}")
            else:
                send_q7(left, gt_left_q7)
                if USE_GRIPPER:
                    set_gripper_ratio(left, gt_left_grip)

            gt_idx += 1

            # ============================================
            # RIGHT ARM: policy prediction (delta-q)
            # ============================================
            if len(rgb_buf) < n_obs_steps:
                # Not enough obs yet, skip policy for right arm
                elapsed = time.perf_counter() - t0
                hz_actual = 1.0 / max(elapsed, 1e-6)
                print(f"\r[{gt_idx}/{gt_total}] {hz_actual:.1f} Hz  (warming up obs buf)", end="", flush=True)
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # Run policy if action chunk exhausted
            if chunk_idx >= len(action_chunk):
                obs_rgb = torch.stack(list(rgb_buf), dim=0).unsqueeze(0).to(device)
                obs_q   = torch.stack(list(q_buf),   dim=0).unsqueeze(0).to(device)

                obs_dict = {"rgb": obs_rgb, "q": obs_q}

                with torch.no_grad():
                    result = policy.predict_action(obs_dict)

                action_chunk = result["action"][0].cpu().numpy()  # (n_action_steps, 16)
                chunk_idx = 0

            # Get current action from chunk
            dq_raw = action_chunk[chunk_idx].astype(np.float32)
            chunk_idx += 1

            # Safety + EMA (right arm only, indices 8:15 for joints, 15 for gripper)
            dq_raw = safe_dq(dq_raw)
            dq_ema = EMA_ALPHA * dq_ema + (1.0 - EMA_ALPHA) * dq_raw

            qR_cmd = qR + dq_ema[8:15]
            gR_cmd = gR + float(dq_ema[15])

            if DRY_RUN:
                print(f"\n[POL R] dqR7={dq_ema[8:15]}  dqRg={dq_ema[15]:+.4f}")
            else:
                send_q7(right, qR_cmd)
                if USE_GRIPPER:
                    set_gripper_ratio(right, gR_cmd)

            # ---- Timing ----
            elapsed = time.perf_counter() - t0
            hz_actual = 1.0 / max(elapsed, 1e-6)
            print(f"\r[{gt_idx}/{gt_total}] {hz_actual:.1f} Hz  (loop {elapsed*1000:.1f} ms)", end="", flush=True)
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
