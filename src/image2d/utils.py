import os
import random
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    x: (3, H, W) in [-1, 1]
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_image_grid(images, out_path, nrow=2):
    """
    images: list of PIL Images
    """
    if len(images) == 0:
        return

    w, h = images[0].size
    ncol = nrow
    nrows = (len(images) + ncol - 1) // ncol

    grid = Image.new("RGB", (ncol * w, nrows * h))

    for idx, img in enumerate(images):
        r = idx // ncol
        c = idx % ncol
        grid.paste(img, (c * w, r * h))

    grid.save(out_path)