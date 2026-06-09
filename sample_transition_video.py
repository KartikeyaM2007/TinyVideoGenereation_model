import argparse
import yaml
import torch
from pathlib import Path

from src.image2d.unet2d import UNet2DClassConditioned
from src.image2d.diffusion2d import GaussianDiffusion2D

from src.transition.model import FrameTransitionUNet
from src.transition.utils import (
    CLASS_TO_ID,
    ensure_dir,
    tensor_to_pil,
    save_gif,
    save_strip,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def generate_first_frame(
    image_cfg_path,
    image_ckpt_path,
    class_name,
    steps,
    guidance,
    device,
):
    image_cfg = load_config(image_cfg_path)

    class_id = CLASS_TO_ID[class_name]

    model = UNet2DClassConditioned(
        in_channels=image_cfg["model"]["in_channels"],
        base_channels=image_cfg["model"]["base_channels"],
        channel_mults=tuple(image_cfg["model"]["channel_mults"]),
        num_classes=image_cfg["model"]["num_classes"],
        time_emb_dim=image_cfg["model"]["time_emb_dim"],
        class_emb_dim=image_cfg["model"]["class_emb_dim"],
        dropout=image_cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(image_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion2D(
        timesteps=image_cfg["diffusion"]["timesteps"],
        beta_start=image_cfg["diffusion"]["beta_start"],
        beta_end=image_cfg["diffusion"]["beta_end"],
        device=device,
    )

    y = torch.tensor([class_id], device=device, dtype=torch.long)

    x = diffusion.sample(
        model,
        shape=(1, 3, image_cfg["data"]["resolution"], image_cfg["data"]["resolution"]),
        y=y,
        sampling_steps=steps,
        guidance_scale=guidance,
    )

    return x


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transition.yaml")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    sample_cfg = cfg["sample"]

    class_name = args.class_name or sample_cfg["class_name"]
    num_frames = args.num_frames or sample_cfg["num_frames"]

    if class_name not in CLASS_TO_ID:
        raise ValueError(f"Unknown class_name: {class_name}")

    class_id = CLASS_TO_ID[class_name]
    y = torch.tensor([class_id], device=device, dtype=torch.long)

    out_dir = Path(sample_cfg["output_dir"])
    ensure_dir(str(out_dir))

    print(f"Generating first frame using image diffusion: {class_name}")

    current = generate_first_frame(
        image_cfg_path=sample_cfg["image_diffusion_config"],
        image_ckpt_path=sample_cfg["image_diffusion_checkpoint"],
        class_name=class_name,
        steps=sample_cfg["image_sampling_steps"],
        guidance=sample_cfg["guidance_scale"],
        device=device,
    )

    transition_model = FrameTransitionUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        num_classes=cfg["model"]["num_classes"],
        class_emb_dim=cfg["model"]["class_emb_dim"],
        dropout=cfg["model"]["dropout"],
        predict_residual=cfg["model"]["predict_residual"],
    ).to(device)

    ckpt = torch.load(sample_cfg["transition_checkpoint"], map_location=device)
    transition_model.load_state_dict(ckpt["model"])
    transition_model.eval()

    frames = [current[0]]

    for _ in range(num_frames - 1):
        next_frame = transition_model(current, y)
        frames.append(next_frame[0])
        current = next_frame

    pil_frames = [tensor_to_pil(frame) for frame in frames]

    gif_path = out_dir / f"{class_name}_transition_{num_frames}f.gif"
    strip_path = out_dir / f"{class_name}_transition_{num_frames}f_strip.png"

    save_gif(pil_frames, str(gif_path), duration=300)
    save_strip(pil_frames, str(strip_path))

    print(f"Saved GIF: {gif_path}")
    print(f"Saved strip: {strip_path}")


if __name__ == "__main__":
    main()