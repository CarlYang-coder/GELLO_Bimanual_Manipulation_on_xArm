import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, random_split

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from gello_video_dataset import GelloVideoDataset
from resnet_video_encoder import ResNet18VideoEncoder


def _collate(batch_list):
    # batch_list: list of dicts from dataset
    # stack to:
    #  obs/rgb: (B,To,3,H,W)
    #  obs/q:   (B,To,16)
    #  action:  (B,T,16)
    rgb = torch.stack([b["obs"]["rgb"] for b in batch_list], dim=0)
    q = torch.stack([b["obs"]["q"] for b in batch_list], dim=0)
    action = torch.stack([b["action"] for b in batch_list], dim=0)
    return {"obs": {"rgb": rgb, "q": q}, "action": action}


def main():
    # -----------------------------
    # HARD-CODED CONFIG (edit here)
    # -----------------------------
    DATA_ROOT = Path(r"D:\Image_DP\data\GelloAgent")  # contains many sessions
    CKPT_DIR = Path(r"D:\Image_DP\ckpts_video_downsampled")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Core horizons
    N_OBS_STEPS = 2          # To
    N_ACTION_STEPS = 8       # chunk length to execute online
    HORIZON = 16             # must be divisible by 4 (UNet downsamples 2x); >= To + Ta

    IMG_SIZE = 128
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50

    # Diffusion timesteps
    NUM_TRAIN_TIMESTEPS = 50
    NUM_INFER_STEPS = 50

    # train/val split
    VAL_RATIO = 0.1
    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    # -----------------------------
    # Build dataset
    # -----------------------------
    csvs = sorted(DATA_ROOT.glob(r"**/joint_with_images_downsampled.csv"))
    assert len(csvs) > 0, f"No joint_with_images_downsampled.csv found under {DATA_ROOT}"

    ds = GelloVideoDataset(
        session_csv_paths=csvs,
        n_obs_steps=N_OBS_STEPS,
        horizon=HORIZON,
        img_size=IMG_SIZE,
        img_subdir="images",
        image_col="image_relpath",   # downsampled rows are non-consecutive; use CSV column to locate images
        stride=1
    )

    n_val = max(1, int(len(ds) * VAL_RATIO))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=_collate
    )

    # -----------------------------
    # Build model parts
    # -----------------------------
    rgb_net = ResNet18VideoEncoder(
        out_dim=512,
        pool="mean",
        mlp_hidden=512,
        dropout=0.0,
        pretrained=True,          # <- pretrained weights
        freeze_backbone=True      # <- freeze backbone, train only MLP head
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
            "q": {"shape": [16], "type": "lowdim"},
        }
    }

    from diffusion_unet_video_policy import DiffusionUnetVideoPolicy as _Policy

    # -----------------------------
    # Build policy (uses original compute_loss from diffusion_policy)
    # -----------------------------
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

    policy = policy.to(DEVICE)

    # -----------------------------
    # Fit normalizer using LinearNormalizer from diffusion_policy
    # Collect q and action data from training set, then call .fit()
    # rgb is already ImageNet-normalized by transforms, no need to fit.
    # -----------------------------
    qs = []
    acts = []
    n = 0
    for batch in train_loader:
        q = batch["obs"]["q"]       # (B,To,16)
        a = batch["action"]         # (B,T,16)
        qs.append(q.reshape(-1, q.shape[-1]))
        acts.append(a.reshape(-1, a.shape[-1]))
        n += 1
        if n >= 100:
            break
    q_all = torch.cat(qs, dim=0)
    a_all = torch.cat(acts, dim=0)

    normalizer = LinearNormalizer()
    normalizer.fit(data={"q": q_all, "action": a_all}, mode="limits", last_n_dims=1)
    # rgb is already ImageNet-normalized by transforms; use identity normalizer
    normalizer["rgb"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    # DictOfTensorMixin._load_from_state_dict clones on CPU; move normalizer to GPU
    policy.normalizer.to(DEVICE)

    # -----------------------------
    # Optimizer: train everything except frozen resnet backbone
    # -----------------------------
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # -----------------------------
    # Train loop
    # -----------------------------
    best_val = float("inf")

    def save_ckpt(path: Path, extra: Dict[str, Any]):
        state = {
            "policy": policy.state_dict(),
            "config": {
                "img_size": IMG_SIZE,
                "n_obs_steps": N_OBS_STEPS,
                "n_action_steps": N_ACTION_STEPS,
                "horizon": HORIZON,
                "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
                "num_infer_steps": NUM_INFER_STEPS,
            }
        }
        state.update(extra)
        torch.save(state, str(path))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        policy.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            # move to device
            batch = {
                "obs": {k: v.to(DEVICE, non_blocking=True) for k, v in batch["obs"].items()},
                "action": batch["action"].to(DEVICE, non_blocking=True),
            }
            opt.zero_grad(set_to_none=True)
            loss = policy.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            train_loss += float(loss.item())
            n_train_batches += 1

        train_loss /= max(1, n_train_batches)

        # validation
        policy.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    "obs": {k: v.to(DEVICE, non_blocking=True) for k, v in batch["obs"].items()},
                    "action": batch["action"].to(DEVICE, non_blocking=True),
                }
                loss = policy.compute_loss(batch)
                val_loss += float(loss.item())
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)

        dt = time.time() - t0
        print(f"[E{epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}  time={dt:.1f}s")

        # save latest
        save_ckpt(CKPT_DIR / "latest.pt", extra={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(CKPT_DIR / "best.pt", extra={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"  [BEST] saved -> {CKPT_DIR / 'best.pt'}")

    print("[DONE] Training finished.")
    print("Best val:", best_val)
    print("CKPT dir:", CKPT_DIR)


if __name__ == "__main__":
    main()
