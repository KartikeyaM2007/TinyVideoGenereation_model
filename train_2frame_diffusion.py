import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.video2f.dataset import FramePairDataset
from src.video2f.utils import set_seed, ensure_dir
from src.image2d.unet2d import UNet2DClassConditioned
from src.image2d.diffusion2d import GaussianDiffusion2D


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/video2f_class.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = FramePairDataset(
        metadata_csv=cfg["data"]["metadata_csv"],
        resolution=cfg["data"]["resolution"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    model = UNet2DClassConditioned(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_classes=cfg["model"]["num_classes"],
        time_emb_dim=cfg["model"]["time_emb_dim"],
        class_emb_dim=cfg["model"]["class_emb_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    diffusion = GaussianDiffusion2D(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg["train"]["mixed_precision"]))

    out_dir = cfg["train"]["output_dir"]
    ensure_dir(out_dir)

    epochs = cfg["train"]["epochs"]
    save_every = cfg["train"]["save_every_epochs"]
    grad_clip = cfg["train"]["grad_clip"]

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        running_loss = 0.0
        step_count = 0

        for batch in pbar:
            x = batch["pair"].to(device)
            y = batch["label_id"].to(device)

            t = torch.randint(
                0,
                cfg["diffusion"]["timesteps"],
                (x.shape[0],),
                device=device,
                dtype=torch.long
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda" and cfg["train"]["mixed_precision"])):
                loss = diffusion.p_losses(model, x, t, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step_count += 1
            avg_loss = running_loss / step_count
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
        }

        latest_path = os.path.join(out_dir, "latest.pt")
        torch.save(ckpt, latest_path)

        if epoch % save_every == 0:
            epoch_path = os.path.join(out_dir, f"epoch_{epoch:03d}.pt")
            torch.save(ckpt, epoch_path)

        print(f"Saved checkpoint: {latest_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()