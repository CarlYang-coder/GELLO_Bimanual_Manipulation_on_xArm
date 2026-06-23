"""
GelloVideoDataset variant that uses **absolute target joint positions**
from the CSV action columns (left_action_*, right_action_*) instead of
computing delta-q.  This aligns with the official Diffusion Policy
convention where actions are absolute target positions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class GelloVideoSample:
    obs: Dict[str, torch.Tensor]
    action: torch.Tensor


def _default_img_transform(img_size: int = 128):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class GelloVideoDatasetAbs(Dataset):
    """Dataset that returns absolute target joint positions as actions."""

    def __init__(
        self,
        session_csv_paths: Union[str, Path, List[Union[str, Path]]],
        n_obs_steps: int,
        horizon: int,
        img_subdir: str = "images",
        img_size: int = 224,
        q_keys: Optional[List[str]] = None,
        action_keys: Optional[List[str]] = None,
        image_col: Optional[str] = None,
        stride: int = 1,
        max_sessions: Optional[int] = None,
        transform=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.img_subdir = img_subdir
        self.img_size = int(img_size)
        self.stride = int(stride)
        self.dtype = dtype

        if transform is None:
            transform = _default_img_transform(self.img_size)
        self.transform = transform

        # Collect CSVs
        csvs: List[Path] = []
        if isinstance(session_csv_paths, (str, Path)):
            p = Path(session_csv_paths)
            if p.is_dir():
                csvs = sorted(p.glob("**/joint_with_images.csv"))
            else:
                csvs = [p]
        else:
            csvs = [Path(x) for x in session_csv_paths]
        csvs = [c for c in csvs if c.exists()]
        if max_sessions is not None:
            csvs = csvs[: int(max_sessions)]
        assert len(csvs) > 0, "No session CSVs found."

        self.sessions = []
        self.index = []

        # Default q keys (16D joint positions)
        if q_keys is None:
            q_keys = (
                [f"left_joint_positions_{i}" for i in range(1, 9)] +
                [f"right_joint_positions_{i}" for i in range(1, 9)]
            )
        self.q_keys = q_keys

        # Default action keys (16D absolute target positions from CSV)
        if action_keys is None:
            action_keys = (
                [f"left_action_{i}" for i in range(1, 9)] +
                [f"right_action_{i}" for i in range(1, 9)]
            )
        self.action_keys = action_keys
        self.image_col = image_col

        for si, csv_path in enumerate(csvs):
            session_dir = csv_path.parent
            img_dir = session_dir / self.img_subdir
            assert img_dir.exists(), f"Image folder not found: {img_dir}"

            df = pd.read_csv(csv_path)
            for k in self.q_keys:
                if k not in df.columns:
                    raise KeyError(f"Missing column '{k}' in {csv_path}")
            for k in self.action_keys:
                if k not in df.columns:
                    raise KeyError(f"Missing action column '{k}' in {csv_path}")

            # Build image paths
            if self.image_col is not None and self.image_col in df.columns:
                img_paths = [session_dir / str(x) for x in df[self.image_col].tolist()]
            else:
                img_paths = sorted([p for p in img_dir.iterdir()
                                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                n = min(len(img_paths), len(df))
                df = df.iloc[:n].reset_index(drop=True)
                img_paths = img_paths[:n]

            q = df[self.q_keys].to_numpy(dtype=np.float32)              # (N, 16)
            action_data = df[self.action_keys].to_numpy(dtype=np.float32)  # (N, 16)

            N = len(df)
            To = self.n_obs_steps
            T = self.horizon
            max_start = N - max(To, T)
            if max_start < 0:
                continue

            self.sessions.append({
                "csv_path": csv_path,
                "session_dir": session_dir,
                "img_paths": img_paths,
                "q": q,            # (N, 16) absolute joint positions
                "action": action_data,  # (N, 16) absolute target positions
                "N": N
            })
            for s in range(0, max_start + 1, self.stride):
                self.index.append((len(self.sessions) - 1, s))

        assert len(self.index) > 0, "No valid training windows (check n_obs_steps/horizon vs sequence lengths)."

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        si, s = self.index[idx]
        sess = self.sessions[si]
        To = self.n_obs_steps
        T = self.horizon

        # Load obs images [s : s+To)
        img_paths = sess["img_paths"][s:s+To]
        imgs = []
        from PIL import Image
        for p in img_paths:
            im = Image.open(p).convert("RGB")
            im = self.transform(im)
            imgs.append(im)
        rgb = torch.stack(imgs, dim=0).to(self.dtype)  # (To, 3, H, W)

        # Lowdim obs: absolute joint positions
        q = torch.from_numpy(sess["q"][s:s+To]).to(self.dtype)  # (To, 16)

        # Action: absolute target positions over horizon [s : s+T)
        action = torch.from_numpy(sess["action"][s:s+T]).to(self.dtype)  # (T, 16)

        return {
            "obs": {
                "rgb": rgb,
                "q": q,
            },
            "action": action
        }

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Official-style normalizer: fit on ALL data from every session.
        Returns a LinearNormalizer with keys: q, action, rgb (identity).
        """
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer, SingleFieldLinearNormalizer,
        )

        all_q = []
        all_action = []
        for sess in self.sessions:
            all_q.append(torch.from_numpy(sess["q"]))
            all_action.append(torch.from_numpy(sess["action"]))

        all_q = torch.cat(all_q, dim=0)
        all_action = torch.cat(all_action, dim=0)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"q": all_q, "action": all_action},
            mode=mode, last_n_dims=1, **kwargs,
        )
        normalizer["rgb"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
