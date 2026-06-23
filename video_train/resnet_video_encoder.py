from __future__ import annotations
from typing import Optional, Literal

import torch
import torch.nn as nn


class ResNet18VideoEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 512,
        pool: Literal["mean", "last"] = "mean",
        mlp_hidden: int = 512,
        dropout: float = 0.0,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.pool = pool

        # torchvision import kept inside to avoid import issues if torchvision not installed
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Remove classification head, keep conv->avgpool->flatten
        # backbone.forward(x) normally returns logits; we want features before fc.
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,  # (B,512,1,1)
        )
        self.backbone_feat_dim = 512

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Trainable head (ONLY this MLP is trainable if freeze_backbone=True)
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(self.backbone_feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, self.out_dim),
        ])
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def _extract_backbone_feat(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,H,W) -> (N,512)
        feat = self.backbone(x)          # (N,512,1,1)
        feat = feat.flatten(1)           # (N,512)
        return feat

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B,T,3,H,W)
        returns: (B,Do)
        """
        assert video.ndim == 5, f"Expected (B,T,C,H,W), got {tuple(video.shape)}"
        B, T, C, H, W = video.shape
        assert C == 3, f"Expected RGB with C=3, got C={C}"

        x = video.reshape(B * T, C, H, W)

        # backbone frozen by requires_grad, but we still allow gradients through head only
        feat = self.backbone(x).flatten(1)          # (B*T,512)
        proj = self.mlp(feat)                       # (B*T,Do)
        proj = proj.reshape(B, T, self.out_dim)     # (B,T,Do)

        if self.pool == "mean":
            out = proj.mean(dim=1)
        elif self.pool == "last":
            out = proj[:, -1]
        else:
            raise ValueError(f"Unknown pool={self.pool}")

        return out