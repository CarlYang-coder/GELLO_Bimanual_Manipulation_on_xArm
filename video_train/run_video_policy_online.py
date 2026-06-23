# -*- coding: utf-8 -*-
r"""
Online rollout for DiffusionUnetVideoPolicy (video obs variant).

Uses the original diffusion_policy predict_action() for inference.
Hardware setup adapted from run_policy_online.py.
Model construction mirrors train_diffusion_video.py exactly.

Behavior:
- Initialize camera + both xArm7
- Press ENTER to start sending commands
- Ctrl+C to stop safely

Act: 16D = [left 7 joints + left gripper, right 7 joints + right gripper]
Obs: rgb (To frames, ImageNet-normalized) + q (16D joint positions)
Action: delta-q (dq = q_next - q_current)
"""

# ============================================================
# HARD-CODED CONFIG
# ============================================================

CKPT_PATH = r"D:\Image_DP\ckpts_video_downsampled\best.pt"

LEFT_ARM_IP  = "192.168.1.224"
RIGHT_ARM_IP = "192.168.1.235"

CAM_INDEX = 0
HZ        = 8.0

# ---- Safety / stability (same as run_policy_online.py) ----
MAX_DQ_JOINT = 0.01   # rad per step (15Hz -> ~0.15 rad/s)
MAX_DQ_GRIP  = 0.02   # gripper ratio per step
EMA_ALPHA    = 0.8     # higher = smoother (0.7~0.9)
DEADBAND_JOINT = 0.001
DEADBAND_GRIP  = 0.002

SPEED = 0.4
MVACC = 1.0

USE_GRIPPER = True

# If True: don't send commands, only print dq/q (recommended first run)
DRY_RUN = False

# ============================================================
# CODE
# ============================================================

import time
import signal
import collections

import cv2
import numpy as np
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

CROP_BOX = (40, 0, 500, 460)  # same crop as Downsampling.py: (left, upper, right, lower) -> 460x460 square

def _make_img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
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
    """xArm pos 0=closed, 850=open -> ratio 1.0=closed, 0.0=open (matches CSV convention)."""
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


# ---------------------------
# Build policy from checkpoint
# (mirrors train_diffusion_video.py exactly)
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

    # ---- Build model parts (same as train_diffusion_video.py) ----
    rgb_net = ResNet18VideoEncoder(
        out_dim=512,
        pool="mean",
        mlp_hidden=512,
        dropout=0.0,
        pretrained=False,        # weights come from checkpoint
        freeze_backbone=True,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        prediction_type="epsilon",
    )

    # shape_meta expected by your policy
    shape_meta = {
        "action": {"shape": [16]},
        "obs": {
            "rgb": {"shape": [3, IMG_SIZE, IMG_SIZE], "type": "rgb"},
            "q":   {"shape": [16], "type": "lowdim"},
        }
    }

    # ---- Build policy (same as train_diffusion_video.py) ----
    policy = _Policy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        rgb_net=rgb_net,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        n_obs_steps=N_OBS_STEPS,
        num_inference_steps=NUM_INFER_STEPS,
        lowdim_as_global_cond=True,
        predict_epsilon=True,
    )

    # Load trained weights (includes normalizer via set_normalizer -> state_dict)
    policy.load_state_dict(ckpt["policy"])
    policy = policy.to(device)
    policy.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"       epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")
    print(f"       horizon={HORIZON}  n_obs={N_OBS_STEPS}  n_act={N_ACTION_STEPS}")
    print(f"       infer_steps={NUM_INFER_STEPS}  img_size={IMG_SIZE}")

    return policy, IMG_SIZE, N_OBS_STEPS, N_ACTION_STEPS


# ---------------------------
# Apply safety filters to delta-q
# ---------------------------

def safe_dq(dq):
    """Clamp, deadband on 16-dim delta-q."""
    dq = dq.copy()
    # left joints [0:7], left grip [7], right joints [8:15], right grip [15]
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

    # ---- Build policy (same construction as train_diffusion_video.py) ----
    policy, img_size, n_obs_steps, n_action_steps = build_policy(CKPT_PATH, device)

    # ---- Image transform (same as gello_video_dataset.py) ----
    img_transform = _make_img_transform(img_size)

    # ---- Camera ----
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    assert cap.isOpened(), "Camera open failed"

    # ---- Arms ----
    left  = init_arm(LEFT_ARM_IP)
    right = init_arm(RIGHT_ARM_IP)

    # ---- Observation ring buffers (To frames, matching dataset format) ----
    # rgb_buf: each element is (3, H, W) tensor (ImageNet-normalized, same as dataset)
    # q_buf:   each element is (16,) tensor
    rgb_buf = collections.deque(maxlen=n_obs_steps)
    q_buf   = collections.deque(maxlen=n_obs_steps)

    # EMA state
    dq_ema = np.zeros(16, dtype=np.float32)

    # Action chunk buffer: execute n_action_steps actions before re-querying policy
    action_chunk = []
    chunk_idx = 0

    print("[READY] Press ENTER to start policy. Ctrl+C to stop.")
    input()

    dt = 1.0 / float(HZ)

    try:
        while True:
            t0 = time.perf_counter()

            # ---- Read camera ----
            ok, frame = cap.read()
            if not ok:
                continue

            # ---- Preprocess image (same as Downsampling.py crop + gello_video_dataset.py transform) ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).crop(CROP_BOX)  # crop to 460x460
            rgb_tensor = img_transform(pil_img)  # Resize to square, ToTensor, ImageNet normalize

            # ---- Read robot state ----
            qL = get_q7(left)
            qR = get_q7(right)
            gL = get_gripper_ratio(left)  if USE_GRIPPER else 0.0
            gR = get_gripper_ratio(right) if USE_GRIPPER else 0.0
            q_now = np.concatenate([qL, [gL], qR, [gR]])  # (16,)
            q_tensor = torch.from_numpy(q_now).float()     # (16,)

            # ---- Push into observation buffers ----
            rgb_buf.append(rgb_tensor)
            q_buf.append(q_tensor)

            # Wait until we have enough obs frames
            if len(rgb_buf) < n_obs_steps:
                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

            # ---- Run policy if action chunk exhausted ----
            if chunk_idx >= len(action_chunk):
                # Build obs_dict matching dataset format:
                #   obs/rgb: (B, To, 3, H, W)
                #   obs/q:   (B, To, 16)
                obs_rgb = torch.stack(list(rgb_buf), dim=0).unsqueeze(0).to(device)  # (1, To, 3, H, W)
                obs_q   = torch.stack(list(q_buf),   dim=0).unsqueeze(0).to(device)  # (1, To, 16)

                obs_dict = {
                    "rgb": obs_rgb,
                    "q":   obs_q,
                }

                # predict_action uses normalizer internally, same as training
                with torch.no_grad():
                    result = policy.predict_action(obs_dict)

                # result['action']: (1, n_action_steps, 16) — delta-q
                action_chunk = result["action"][0].cpu().numpy()  # (n_action_steps, 16)
                chunk_idx = 0

            # ---- Get current action from chunk ----
            dq_raw = action_chunk[chunk_idx].astype(np.float32)
            chunk_idx += 1

            # ---- Safety ----
            dq_raw = safe_dq(dq_raw)

            # ---- EMA smooth ----
            dq_ema = EMA_ALPHA * dq_ema + (1.0 - EMA_ALPHA) * dq_raw

            # ---- Compute target joint positions: q_target = q_current + dq ----
            qL_cmd = qL + dq_ema[0:7]
            qR_cmd = qR + dq_ema[8:15]
            gL_cmd = gL + float(dq_ema[7])
            gR_cmd = gR + float(dq_ema[15])

            # ---- Send ----
            if DRY_RUN:
                print(f"dqL7={dq_ema[0:7]} dqLg={dq_ema[7]:+.4f} | "
                      f"dqR7={dq_ema[8:15]} dqRg={dq_ema[15]:+.4f}")
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
