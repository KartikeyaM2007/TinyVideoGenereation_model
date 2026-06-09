import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.transition.dataset import FrameTransitionDataset
from src.transition.model import FrameTransitionUNet
from src.transition.utils import set_seed, ensure_dir


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def temporal_difference_loss(current_frame, pred_next, real_next):
    """
    Forces the predicted frame to learn actual change, not just copy the input.
    """
    real_delta = real_next - current_frame
    pred_delta = pred_next - current_frame

    delta_loss = F.l1_loss(pred_delta, real_delta)

    real_motion_mag = real_delta.abs().mean(dim=[1, 2, 3])
    pred_motion_mag = pred_delta.abs().mean(dim=[1, 2, 3])
    motion_mag_loss = F.l1_loss(pred_motion_mag, real_motion_mag)

    return delta_loss + 2.0 * motion_mag_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transition.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = FrameTransitionDataset(
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

    model = FrameTransitionUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        num_classes=cfg["model"]["num_classes"],
        class_emb_dim=cfg["model"]["class_emb_dim"],
        dropout=cfg["model"]["dropout"],
        predict_residual=cfg["model"]["predict_residual"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda" and cfg["train"]["mixed_precision"]),
    )

    out_dir = cfg["train"]["output_dir"]
    ensure_dir(out_dir)

    epochs = cfg["train"]["epochs"]
    save_every = cfg["train"]["save_every_epochs"]

    l1_weight = cfg["train"]["l1_weight"]
    mse_weight = cfg["train"]["mse_weight"]
    temporal_weight = cfg["train"]["temporal_weight"]
    grad_clip = cfg["train"]["grad_clip"]

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")

        running_loss = 0.0
        step_count = 0

        for batch in pbar:
            current_frame = batch["current_frame"].to(device)
            next_frame = batch["next_frame"].to(device)
            label_id = batch["label_id"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda" and cfg["train"]["mixed_precision"]),
            ):
                pred_next = model(current_frame, label_id)

                l1 = F.l1_loss(pred_next, next_frame)
                mse = F.mse_loss(pred_next, next_frame)
                temporal = temporal_difference_loss(current_frame, pred_next, next_frame)

                loss = (
                    l1_weight * l1
                    + mse_weight * mse
                    + temporal_weight * temporal
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step_count += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                l1=f"{l1.item():.4f}",
                temp=f"{temporal.item():.4f}",
                avg=f"{running_loss / step_count:.4f}",
            )

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