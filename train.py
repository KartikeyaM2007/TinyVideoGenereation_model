from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VideoPromptDataset
from src.diffusion.gaussian import GaussianDiffusion
from src.models.unet3d import TinyVideoUNet
from src.utils.config import get_device, load_config, seed_everything


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = VideoPromptDataset(cfg["data"]["metadata_csv"], cfg["data"]["embeddings_path"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model_cfg = cfg["model"]
    model = TinyVideoUNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=list(model_cfg["channel_mults"]),
        text_dim=int(cfg["text"]["embedding_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_start=float(cfg["diffusion"]["beta_start"]),
        beta_end=float(cfg["diffusion"]["beta_end"]),
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1e-4)
    use_amp = bool(cfg["train"]["mixed_precision"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    output_dir = Path(cfg["train"]["output_dir"])
    grad_accum = int(cfg["train"]["grad_accum_steps"])
    cond_drop_prob = float(cfg["train"]["cond_drop_prob"])
    global_step = 0

    model.train()
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(pbar, start=1):
            video = batch["video"].to(device, non_blocking=True)  # B,C,T,H,W
            text_emb = batch["text_emb"].to(device, non_blocking=True)

            # classifier-free conditioning dropout
            if cond_drop_prob > 0:
                keep = (torch.rand(text_emb.shape[0], device=device) > cond_drop_prob).float().view(-1, 1)
                text_emb = text_emb * keep

            t = torch.randint(0, diffusion.timesteps, (video.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(video)
            noisy_video = diffusion.q_sample(video, t, noise)

            with autocast(enabled=use_amp):
                pred_noise = model(noisy_video, t, text_emb)
                loss = F.mse_loss(pred_noise, noise) / grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * grad_accum

            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix(loss=running_loss / step, step=global_step)

        if epoch % int(cfg["train"]["save_every_epochs"]) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg)
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, cfg)
            print(f"Saved checkpoint: {output_dir / 'latest.pt'}")

    print("Training complete.")


if __name__ == "__main__":
    main()
