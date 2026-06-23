"""
Official-style workspace-based training for DiffusionUnetVideoPolicy.

Follows the pattern from diffusion_policy/workspace/train_diffusion_unet_video_workspace.py:
  - BaseWorkspace-style class with save/load checkpoint, EMA, LR scheduler, WandB logging
  - TopKCheckpointManager for best-k checkpoints
  - Gradient accumulation
  - Validation via predict_action MSE

Usage:
    python train_DP_OfficialVideo.py
"""

import os
import sys
import copy
import time
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# ---- diffusion_policy library imports ----
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

# ---- local imports ----
from gello_video_dataset_abs import GelloVideoDatasetAbs as GelloVideoDataset
from resnet_video_encoder import ResNet18VideoEncoder
from diffusion_unet_video_policy import DiffusionUnetVideoPolicy as _Policy


# ============================================================
# CONFIG  (edit this section — mirrors the role of Hydra YAML)
# ============================================================
cfg = dict(
    # ---- data ----
    data_root       = r"D:\Image_DP\data\GelloAgent",
    csv_pattern     = "**/joint_with_images_downsampled.csv",
    img_subdir      = "images",
    image_col       = "image_relpath",

    # ---- model / horizons ----
    img_size        = 128,
    n_obs_steps     = 2,
    n_action_steps  = 8,
    horizon         = 16,       # must be divisible by 4; >= n_obs_steps + n_action_steps
    action_dim      = 16,

    # ---- diffusion ----
    num_train_timesteps = 50,
    num_infer_steps     = 50,
    beta_schedule       = "squaredcos_cap_v2",
    prediction_type     = "epsilon",
    predict_epsilon     = True,

    # ---- encoder ----
    rgb_out_dim       = 512,
    rgb_pool          = "mean",
    rgb_mlp_hidden    = 512,
    rgb_dropout       = 0.0,
    rgb_pretrained    = True,
    rgb_freeze_backbone = True,

    # ---- training ----
    seed              = 42,
    num_epochs        = 8000,
    batch_size        = 64,
    num_workers       = 2,
    lr                = 1e-4,
    weight_decay      = 1e-4,
    max_grad_norm     = 1.0,
    gradient_accumulate_every = 1,
    val_ratio         = 0.1,

    # ---- lr scheduler ----
    lr_scheduler      = "cosine",   # "cosine", "constant", "constant_with_warmup", etc.
    lr_warmup_steps   = 500,

    # ---- EMA ----
    use_ema           = True,
    ema_inv_gamma     = 1.0,
    ema_power         = 0.75,
    ema_max_value     = 0.9999,

    # ---- logging ----
    use_wandb         = False,       # set True to enable WandB
    wandb_project     = "diffusion_policy_video",
    wandb_name        = None,        # auto-generated if None

    # ---- checkpointing ----
    output_dir        = r"D:\Image_DP\ckpts_official_video",
    save_last_ckpt    = True,
    topk_k            = 5,
    topk_monitor_key  = "val_loss",
    topk_mode         = "min",

    # ---- validation (predict_action MSE) ----
    val_every         = 50,         # global steps

    # ---- tqdm ----
    tqdm_interval_sec = 5.0,

    # ---- device ----
    device            = "cuda",
)


# ============================================================
# COLLATE
# ============================================================
def _collate(batch_list):
    rgb    = torch.stack([b["obs"]["rgb"] for b in batch_list], dim=0)
    q      = torch.stack([b["obs"]["q"]   for b in batch_list], dim=0)
    action = torch.stack([b["action"]     for b in batch_list], dim=0)
    return {"obs": {"rgb": rgb, "q": q}, "action": action}


# ============================================================
# WORKSPACE
# ============================================================
class TrainDiffusionUnetVideoWorkspace:
    """
    Official-style workspace:
      - EMA model
      - LR scheduler with warmup
      - Gradient accumulation
      - WandB logging
      - TopK + last checkpoint saving
      - Validation via predict_action MSE
      - Resume from checkpoint
    """

    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---- seed ----
        seed = cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ---- build policy ----
        self.model = self._build_policy()

        # ---- EMA ----
        self.ema_model = None
        if cfg["use_ema"]:
            self.ema_model = copy.deepcopy(self.model)

        # ---- optimizer ----
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # ---- training state ----
        self.global_step = 0
        self.epoch = 0

    # ----------------------------------------------------------
    def _build_policy(self):
        c = self.cfg
        rgb_net = ResNet18VideoEncoder(
            out_dim=c["rgb_out_dim"],
            pool=c["rgb_pool"],
            mlp_hidden=c["rgb_mlp_hidden"],
            dropout=c["rgb_dropout"],
            pretrained=c["rgb_pretrained"],
            freeze_backbone=c["rgb_freeze_backbone"],
        )
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=c["num_train_timesteps"],
            beta_schedule=c["beta_schedule"],
            clip_sample=False,
            prediction_type=c["prediction_type"],
        )
        shape_meta = {
            "action": {"shape": [c["action_dim"]]},
            "obs": {
                "rgb": {"shape": [3, c["img_size"], c["img_size"]], "type": "rgb"},
                "q":   {"shape": [c["action_dim"]], "type": "lowdim"},
            }
        }
        policy = _Policy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            rgb_net=rgb_net,
            horizon=c["horizon"],
            n_action_steps=c["n_action_steps"],
            n_obs_steps=c["n_obs_steps"],
            num_inference_steps=c["num_infer_steps"],
            lowdim_as_global_cond=True,
            predict_epsilon=c["predict_epsilon"],
        )
        return policy

    # ----------------------------------------------------------
    # Checkpoint helpers
    # ----------------------------------------------------------
    def _state_dict(self):
        """Gather full state for checkpointing."""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "cfg": self.cfg,
        }
        if self.ema_model is not None:
            state["ema_model"] = self.ema_model.state_dict()
        return state

    def save_checkpoint(self, path=None, extra=None):
        if path is None:
            path = str(self.output_dir / "latest.ckpt")
        state = self._state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0)
        if self.ema_model is not None and "ema_model" in state:
            self.ema_model.load_state_dict(state["ema_model"])
        print(f"[RESUME] Loaded checkpoint from {path}  "
              f"(epoch={self.epoch}, global_step={self.global_step})")

    # ----------------------------------------------------------
    # Also save a lightweight inference-only checkpoint
    # compatible with existing run_video_policy_online*.py
    # ----------------------------------------------------------
    def _save_inference_ckpt(self, path, epoch, train_loss, val_loss):
        c = self.cfg
        policy = self.ema_model if self.ema_model is not None else self.model
        state = {
            "policy": policy.state_dict(),
            "config": {
                "img_size": c["img_size"],
                "n_obs_steps": c["n_obs_steps"],
                "n_action_steps": c["n_action_steps"],
                "horizon": c["horizon"],
                "num_train_timesteps": c["num_train_timesteps"],
                "num_infer_steps": c["num_infer_steps"],
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, str(path))

    # ----------------------------------------------------------
    # RUN
    # ----------------------------------------------------------
    def run(self):
        c = self.cfg

        # ---- resume ----
        resume_path = self.output_dir / "latest.ckpt"
        if resume_path.is_file():
            self.load_checkpoint(str(resume_path))

        # ---- dataset ----
        data_root = Path(c["data_root"])
        csvs = sorted(data_root.glob(c["csv_pattern"]))
        assert len(csvs) > 0, f"No CSVs found under {data_root}"

        ds = GelloVideoDataset(
            session_csv_paths=csvs,
            n_obs_steps=c["n_obs_steps"],
            horizon=c["horizon"],
            img_size=c["img_size"],
            img_subdir=c["img_subdir"],
            image_col=c["image_col"],
            stride=1,
        )
        n_val = max(1, int(len(ds) * c["val_ratio"]))
        n_train = len(ds) - n_val
        train_ds, val_ds = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(c["seed"]))

        train_loader = DataLoader(
            train_ds, batch_size=c["batch_size"], shuffle=True,
            num_workers=c["num_workers"], pin_memory=True, collate_fn=_collate)
        val_loader = DataLoader(
            val_ds, batch_size=c["batch_size"], shuffle=False,
            num_workers=c["num_workers"], pin_memory=True, collate_fn=_collate)

        # ---- normalizer (official style: full dataset statistics) ----
        print("[INFO] Fitting normalizer on full dataset ...")
        normalizer = ds.get_normalizer(mode="limits")

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # ---- LR scheduler ----
        lr_scheduler = get_scheduler(
            c["lr_scheduler"],
            optimizer=self.optimizer,
            num_warmup_steps=c["lr_warmup_steps"],
            num_training_steps=(
                len(train_loader) * c["num_epochs"]
            ) // c["gradient_accumulate_every"],
            last_epoch=self.global_step - 1,
        )

        # ---- EMA ----
        ema = None
        if c["use_ema"]:
            ema = EMAModel(
                model=self.ema_model,
                inv_gamma=c["ema_inv_gamma"],
                power=c["ema_power"],
                max_value=c["ema_max_value"],
            )

        # ---- TopK checkpoint manager ----
        topk_manager = TopKCheckpointManager(
            save_dir=str(self.ckpt_dir),
            monitor_key=c["topk_monitor_key"],
            mode=c["topk_mode"],
            k=c["topk_k"],
            format_str="epoch={epoch:03d}-val_loss={val_loss:.6f}.ckpt",
        )

        # ---- WandB ----
        wandb_run = None
        if c["use_wandb"]:
            import wandb
            wandb_run = wandb.init(
                project=c["wandb_project"],
                name=c["wandb_name"],
                dir=str(self.output_dir),
                config=c,
            )

        # ---- device ----
        device = torch.device(c["device"] if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # ---- grab one train batch for predict_action MSE (official style) ----
        val_batch = next(iter(train_loader))

        # ---- training loop ----
        print(f"[INFO] Starting training: {c['num_epochs']} epochs, "
              f"{len(train_loader)} batches/epoch, device={device}")

        best_val = float("inf")

        for _ in range(self.epoch, c["num_epochs"]):
            epoch_t0 = time.time()
            epoch_train_loss = 0.0
            n_batches = 0

            with tqdm.tqdm(
                train_loader,
                desc=f"Epoch {self.epoch}",
                leave=False,
                mininterval=c["tqdm_interval_sec"],
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # ---- device transfer ----
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    # ---- forward ----
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / c["gradient_accumulate_every"]
                    loss.backward()

                    # ---- optimizer step (with gradient accumulation) ----
                    if self.global_step % c["gradient_accumulate_every"] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            c["max_grad_norm"])
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # ---- EMA update ----
                    if ema is not None:
                        ema.step(self.model)

                    # ---- logging ----
                    raw_loss_cpu = raw_loss.item()
                    epoch_train_loss += raw_loss_cpu
                    n_batches += 1

                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }

                    # ---- validation (predict_action MSE) ----
                    if self.global_step > 0 and self.global_step % c["val_every"] == 0:
                        policy = self.ema_model if self.ema_model is not None else self.model
                        policy.eval()
                        with torch.no_grad():
                            vb = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = vb["obs"]
                            gt_action = vb["action"]
                            result = policy.predict_action(obs_dict)
                            pred_action = result["action_pred"]
                            mse = F.mse_loss(pred_action, gt_action)
                            step_log["val_action_mse"] = mse.item()
                        policy.train()

                    # ---- wandb ----
                    if wandb_run is not None:
                        wandb_run.log(step_log, step=self.global_step)

                    self.global_step += 1

            # ---- end of epoch ----
            epoch_train_loss /= max(1, n_batches)

            # ---- full validation loss ----
            val_loss = 0.0
            n_val = 0
            policy = self.ema_model if self.ema_model is not None else self.model
            policy.eval()
            with torch.no_grad():
                for vb in val_loader:
                    vb = dict_apply(vb, lambda x: x.to(device, non_blocking=True))
                    vl = policy.compute_loss(vb)
                    val_loss += vl.item()
                    n_val += 1
            val_loss /= max(1, n_val)
            policy.train()

            dt = time.time() - epoch_t0
            print(f"[E{self.epoch:03d}] train={epoch_train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={lr_scheduler.get_last_lr()[0]:.2e}  "
                  f"time={dt:.1f}s")

            if wandb_run is not None:
                wandb_run.log({
                    "epoch_train_loss": epoch_train_loss,
                    "epoch_val_loss": val_loss,
                    "epoch": self.epoch,
                }, step=self.global_step)

            # ---- save latest full checkpoint (for resume) ----
            if c["save_last_ckpt"]:
                self.save_checkpoint()

            # ---- save inference-compatible checkpoint ----
            self._save_inference_ckpt(
                self.output_dir / "latest.pt",
                epoch=self.epoch, train_loss=epoch_train_loss, val_loss=val_loss)

            if val_loss < best_val:
                best_val = val_loss
                self._save_inference_ckpt(
                    self.output_dir / "best.pt",
                    epoch=self.epoch, train_loss=epoch_train_loss, val_loss=val_loss)
                print(f"  [BEST] val_loss={val_loss:.6f} -> {self.output_dir / 'best.pt'}")

            # ---- TopK checkpoint ----
            topk_path = topk_manager.get_ckpt_path({
                "epoch": self.epoch,
                "val_loss": val_loss,
            })
            if topk_path is not None:
                self.save_checkpoint(path=topk_path)

            self.epoch += 1

        print(f"\n[DONE] Training finished.  Best val_loss={best_val:.6f}")
        print(f"       Output dir: {self.output_dir}")


# ============================================================
# MAIN
# ============================================================
def main():
    workspace = TrainDiffusionUnetVideoWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
