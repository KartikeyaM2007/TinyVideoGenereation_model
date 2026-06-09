import torch
import torch.nn as nn
import torch.nn.functional as F


def make_gn(channels):
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CondResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.05):
        super().__init__()

        self.norm1 = make_gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = make_gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
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
        self.conv = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class FrameTransitionUNet(nn.Module):
    """
    Predicts next frame from current frame + class label.

    Input:
        x: (B, 3, H, W)
        y: (B,)

    Output:
        predicted next frame: (B, 3, H, W), range roughly [-1, 1]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=48,
        num_classes=3,
        class_emb_dim=128,
        dropout=0.05,
        predict_residual=True,
    ):
        super().__init__()

        self.predict_residual = predict_residual

        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, class_emb_dim),
            nn.SiLU(),
            nn.Linear(class_emb_dim, class_emb_dim),
        )

        c = base_channels

        self.in_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        self.down1 = CondResBlock(c, c, class_emb_dim, dropout)
        self.ds1 = Downsample(c)

        self.down2 = CondResBlock(c, c * 2, class_emb_dim, dropout)
        self.ds2 = Downsample(c * 2)

        self.down3 = CondResBlock(c * 2, c * 4, class_emb_dim, dropout)

        self.mid1 = CondResBlock(c * 4, c * 4, class_emb_dim, dropout)
        self.mid2 = CondResBlock(c * 4, c * 4, class_emb_dim, dropout)

        self.us2 = Upsample(c * 4)
        self.up2 = CondResBlock(c * 4 + c * 2, c * 2, class_emb_dim, dropout)

        self.us1 = Upsample(c * 2)
        self.up1 = CondResBlock(c * 2 + c, c, class_emb_dim, dropout)

        self.out_norm = make_gn(c)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        emb = self.class_mlp(self.class_emb(y))

        x_in = x

        x0 = self.in_conv(x)

        d1 = self.down1(x0, emb)
        x = self.ds1(d1)

        d2 = self.down2(x, emb)
        x = self.ds2(d2)

        x = self.down3(x, emb)

        x = self.mid1(x, emb)
        x = self.mid2(x, emb)

        x = self.us2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2(x, emb)

        x = self.us1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up1(x, emb)

        out = self.out_conv(F.silu(self.out_norm(x)))

        if self.predict_residual:
            out = x_in + 0.5 * torch.tanh(out)
        else:
            out = torch.tanh(out)

        return out.clamp(-1.0, 1.0)