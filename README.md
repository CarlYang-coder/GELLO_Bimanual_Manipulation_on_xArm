# GELLO Bimanual Manipulation on xArm

Based on [**GELLO**](https://github.com/wuphilipp/gello_software), we present a complete workflow encompassing demonstration collection of bimanual tasks, synchronized logging and dataset curation, BC-based policy training, and deployment on a real robot.

The project focuses on:
- The form of collected data of Bimanual Teleoperation.
- One fixed 3rd view camera with image observation.
- A light weight model of Diffusion Policy.
- The script of real-time inference.

---

## Repository layout

All training / inference code lives under [`video_train/`](video_train):

| File | Purpose |
| --- | --- |
| `train_DP_OfficialVideo.py` | Train the Diffusion Policy (`DiffusionUnetVideoPolicy`, ResNet-18 video encoder + 1D U-Net diffusion). |
| `downsample_and_train.py` | Combined pipeline: downsample/crop the raw demos **and** train, in one script. |
| `gen_cropped_video.py` | Sanity-check tool: renders what the model actually sees (cropped, resized to 128x128) as an mp4. |
| `run_DP_Official_noclamp.py` | Online real-time inference on the dual xArm. No joint-limit clamping. |
| `run_DP_Official.py` | Same as above but clamps each commanded joint to the training data's min/max range (safer). |
| `gello_video_dataset_abs.py` | Dataset (absolute action targets) used by training. |
| `resnet_video_encoder.py` / `diffusion_unet_video_policy.py` | Model definition. |
| `diffusion_policy/` | Vendored subset of the [diffusion_policy](https://github.com/real-stanford/diffusion_policy) library (normalizer, EMA, schedulers, etc.). |

---

## Requirements

- Python 3.10+ with CUDA-enabled PyTorch (an NVIDIA GPU is strongly recommended for training).
- `pip install torch torchvision diffusers numpy pandas opencv-python pillow tqdm`
- For real-robot inference: the **xArm Python SDK** (`pip install xarm-python-sdk`) and two UFACTORY xArm arms reachable on the network.

> **Note:** the scripts use **hard-coded paths and config** (e.g. `DATA_ROOT = r"D:\Image_DP\data\GelloAgent"`, checkpoint dir `D:\Image_DP\ckpts_official_video`, robot IPs, camera index). Open the `CONFIG` / constants block at the top of each script and edit these to match your machine before running.

---

## How to run

The end-to-end pipeline is **collect -> downsample -> (check) -> train -> deploy**.

### 1. Prepare / downsample the data
Raw demonstrations are recorded at 30 Hz. Downsample to 10 Hz and crop the images. This is built into the combined pipeline (`downsample_and_train.py`), or you can run your `Downsampling.py` step separately. After this you should have, per session, a `joint_with_images_downsampled.csv` plus the cropped `images/` folder under `DATA_ROOT`.

### 2. (Optional) Verify what the model sees
```bash
cd video_train
python gen_cropped_video.py                       # all sessions
python gen_cropped_video.py --session 0212_193402 # a single session
```
This writes an mp4 of the 128x128 cropped observation so you can confirm the framing / crop box.

### 3. Train the policy
```bash
cd video_train
python train_DP_OfficialVideo.py
```
- Edit the `cfg` dict at the top of the file first (`data_root`, `output_dir`, `num_epochs`, `batch_size`, `horizon`, etc.).
- Checkpoints are written to `output_dir` (default `D:\Image_DP\ckpts_official_video`):
  - `best.pt` / `latest.pt` - lightweight inference checkpoints
  - `latest.ckpt` - full checkpoint (optimizer + EMA) for resuming training
- Training auto-resumes from `latest.ckpt` if it exists in `output_dir`.

Alternatively, run the **combined** downsample + train pipeline in one go:
```bash
python downsample_and_train.py
```

### 4. Deploy - real-time online inference
Make sure both arms are powered, connected, and the camera is plugged in. Set the robot IPs (`LEFT_ARM_IP` / `RIGHT_ARM_IP`), `CAM_INDEX`, and `CKPT_PATH` at the top of the run script, then:
```bash
cd video_train
python run_DP_Official_noclamp.py
```
The script will:
1. Load `best.pt` and move both arms to the mean initial pose from the training CSVs (press **ENTER** when prompted).
2. Press **ENTER** again to start the policy; it runs at ~10 Hz with EMA smoothing and receding-horizon action chunks. Press **Ctrl+C** to stop.

> **Safety:** `run_DP_Official_noclamp.py` sends the policy output directly (no joint-limit clamping). For a safer first deployment use `run_DP_Official.py`, which clamps every commanded joint to the training data's `Q_MIN`/`Q_MAX` range. You can also set `DRY_RUN = True` at the top of either script to print commands without moving the robot.
