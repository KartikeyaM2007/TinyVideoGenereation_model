import argparse
import yaml
import torch
from pathlib import Path

from src.image2d.unet2d import UNet2DClassConditioned
from src.image2d.diffusion2d import GaussianDiffusion2D
from src.video4f.utils import tensor_to_pil, save_gif, save_strip, ensure_dir


CLASS_TO_ID = {
    "Normal": 0,
    "Fighting": 1,
    "RoadAccidents": 2,
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/video4f_class.yaml")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_cfg = cfg["sample"]

    class_name = args.class_name or sample_cfg["class_name"]
    steps = args.steps or sample_cfg["sampling_steps"]
    guidance = args.guidance or sample_cfg["guidance_scale"]
    output_dir = Path(sample_cfg["output_dir"])

    ensure_dir(str(output_dir))

    if class_name not in CLASS_TO_ID:
        raise ValueError(f"Unknown class_name: {class_name}")

    model = UNet2DClassConditioned(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_classes=cfg["model"]["num_classes"],
        time_emb_dim=cfg["model"]["time_emb_dim"],
        class_emb_dim=cfg["model"]["class_emb_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(sample_cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion2D(
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=device,
    )

    num_samples = sample_cfg["num_samples"]
    y = torch.tensor([CLASS_TO_ID[class_name]] * num_samples, device=device, dtype=torch.long)

    x = diffusion.sample(
        model,
        shape=(num_samples, 12, cfg["data"]["resolution"], cfg["data"]["resolution"]),
        y=y,
        sampling_steps=steps,
        guidance_scale=guidance,
    )

    for i in range(num_samples):
        sample = x[i]
        frames = [
            tensor_to_pil(sample[0:3]),
            tensor_to_pil(sample[3:6]),
            tensor_to_pil(sample[6:9]),
            tensor_to_pil(sample[9:12]),
        ]

        gif_path = output_dir / f"{class_name}_sample_{i+1}.gif"
        strip_path = output_dir / f"{class_name}_sample_{i+1}_strip.png"

        save_gif(frames, str(gif_path), duration=280)
        save_strip(frames, str(strip_path))

    print(f"Saved {num_samples} samples to {output_dir}")


if __name__ == "__main__":
    main()