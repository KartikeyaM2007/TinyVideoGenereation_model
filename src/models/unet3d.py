from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return x * (1 + scale) + shift


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        return self.conv(x)


class TinyVideoUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        channel_mults: List[int] | None = None,
        text_dim: int = 384,
        time_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]

        self.time_dim = time_dim
        cond_dim = time_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        skips = []
        for out_ch in channels:
            self.down_blocks.append(ResBlock3D(ch, out_ch, cond_dim, dropout))
            self.downsamples.append(Downsample3D(out_ch))
            skips.append(out_ch)
            ch = out_ch

        self.mid1 = ResBlock3D(ch, ch, cond_dim, dropout)
        self.mid2 = ResBlock3D(ch, ch, cond_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for skip_ch in reversed(skips):
            self.upsamples.append(Upsample3D(ch))
            self.up_blocks.append(ResBlock3D(ch + skip_ch, skip_ch, cond_dim, dropout))
            ch = skip_ch

        self.out_norm = nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
        self.out_conv = nn.Conv3d(ch, in_channels, kernel_size=3, padding=1)

    def make_cond(self, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_proj(t_emb)
        txt = self.text_proj(text_emb)
        return t_emb + txt

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        cond = self.make_cond(t, text_emb)
        h = self.init_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        for up, block in zip(self.upsamples, self.up_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
