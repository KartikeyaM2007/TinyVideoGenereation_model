import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device).float() / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def make_gn(channels):
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = make_gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = make_gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        emb_out = self.emb_proj(F.silu(emb))
        h = h + emb_out[:, :, None, None]

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet2DClassConditioned(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_classes=3,
        time_emb_dim=128,
        class_emb_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        emb_dim = time_emb_dim + class_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        chs = [base_channels * m for m in channel_mults]

        self.in_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down1 = ResBlock(chs[0], chs[0], emb_dim, dropout)
        self.ds1 = Downsample(chs[0])

        self.down2 = ResBlock(chs[0], chs[1], emb_dim, dropout)
        self.ds2 = Downsample(chs[1])

        self.mid1 = ResBlock(chs[1], chs[2], emb_dim, dropout)
        self.mid2 = ResBlock(chs[2], chs[2], emb_dim, dropout)

        self.us2 = Upsample(chs[2])
        self.up2 = ResBlock(chs[2] + chs[1], chs[1], emb_dim, dropout)

        self.us1 = Upsample(chs[1])
        self.up1 = ResBlock(chs[1] + chs[0], chs[0], emb_dim, dropout)

        self.out_norm = make_gn(chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_channels, 3, padding=1)

    def forward(self, x, t, y):
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        c_emb = self.class_emb(y)
        emb = torch.cat([t_emb, c_emb], dim=-1)

        x0 = self.in_conv(x)

        d1 = self.down1(x0, emb)
        x = self.ds1(d1)

        d2 = self.down2(x, emb)
        x = self.ds2(d2)

        x = self.mid1(x, emb)
        x = self.mid2(x, emb)

        x = self.us2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2(x, emb)

        x = self.us1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up1(x, emb)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x