"""
TemporalAggregator: 1D conv encoder that aggregates a short temporal sequence
of low-dim observations into a single feature vector.

Input:  (B, T, C)   e.g. (B, 2, 16)
Output: (B, D)       flattened feature
"""

import torch
import torch.nn as nn


class _ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, n_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.GroupNorm(min(n_groups, out_ch), out_ch),
            nn.Mish(),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)


class TemporalAggregator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channel_mults=(1, 1),
        n_blocks_per_level: int = 1,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        channels = [in_channels] + [in_channels * m for m in channel_mults]
        layers = []
        for i in range(len(channel_mults)):
            ch_in = channels[i]
            ch_out = channels[i + 1]
            for b in range(n_blocks_per_level):
                layers.append(
                    _ResBlock1d(
                        ch_in if b == 0 else ch_out,
                        ch_out,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                    )
                )
        self.net = nn.Sequential(*layers)
        self._out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) -> (B, D)
        """
        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.net(x)           # (B, C_out, T)
        x = x.flatten(1)         # (B, C_out * T)
        return x
