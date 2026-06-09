from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.data.video_utils import save_gif
from src.diffusion.gaussian import GaussianDiffusion
from src.models.unet3d import TinyVideoUNet
from src.utils.config import get_device, load_config, seed_everything


def encode_prompt(prompt: str, encoder_name: str) -> torch.Tensor:
    model = SentenceTransformer(encoder_name, device="cpu")
    emb = model.encode(prompt, convert_to_tensor=True, normalize_embeddings=True).float().unsqueeze(0)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device()

    prompt = args.prompt or cfg["sample"]["prompt"]
    out_path = Path(args.output or cfg["sample"]["output"])
    ckpt_path = Path(args.checkpoint or cfg["sample"]["checkpoint"])
    sampling_steps = int(args.steps or cfg["sample"]["sampling_steps"])
    guidance = float(args.guidance or cfg["sample"]["guidance_scale"])

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model_cfg = cfg["model"]
    model = TinyVideoUNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=list(model_cfg["channel_mults"]),
        text_dim=int(cfg["text"]["embedding_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"]["timesteps"]),
        beta_start=float(cfg["diffusion"]["beta_start"]),
        beta_end=float(cfg["diffusion"]["beta_end"]),
        device=device,
    )

    text_emb = encode_prompt(prompt, cfg["text"]["encoder_name"]).to(device)
    shape = (
        1,
        int(cfg["model"]["in_channels"]),
        int(cfg["data"]["num_frames"]),
        int(cfg["data"]["resolution"]),
        int(cfg["data"]["resolution"]),
    )

    print(f"Sampling prompt: {prompt}")
    with torch.no_grad():
        video = diffusion.sample(
            model,
            shape=shape,
            text_emb=text_emb,
            sampling_steps=sampling_steps,
            guidance_scale=guidance,
        )[0]

    save_gif(video, out_path, fps=6)
    print(f"Saved GIF: {out_path}")


if __name__ == "__main__":
    main()
