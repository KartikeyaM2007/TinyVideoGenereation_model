import argparse
import yaml
import torch

from src.image2d.unet2d import UNet2DClassConditioned
from src.image2d.diffusion2d import GaussianDiffusion2D
from src.image2d.utils import tensor_to_pil, save_image_grid


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
    parser.add_argument("--config", type=str, default="configs/image2d_class.yaml")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_cfg = cfg["sample"]

    class_name = args.class_name or sample_cfg["class_name"]
    output = args.output or sample_cfg["output"]
    steps = args.steps or sample_cfg["sampling_steps"]
    guidance = args.guidance or sample_cfg["guidance_scale"]

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
        shape=(num_samples, 3, cfg["data"]["resolution"], cfg["data"]["resolution"]),
        y=y,
        sampling_steps=steps,
        guidance_scale=guidance,
    )

    images = [tensor_to_pil(x[i]) for i in range(num_samples)]
    save_image_grid(images, output, nrow=2)
    print(f"Saved samples to {output}")


if __name__ == "__main__":
    main()