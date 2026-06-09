import os
import random
import numpy as np
import torch
from PIL import Image


CLASS_TO_ID = {
    "Normal": 0,
    "Fighting": 1,
    "RoadAccidents": 2,
}

ID_TO_CLASS = {
    0: "Normal",
    1: "Fighting",
    2: "RoadAccidents",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    x: (3, H, W), range [-1, 1]
    """
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    x = (x * 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_gif(frames, out_path, duration=300):
    if not frames:
        return
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def save_strip(frames, out_path):
    if not frames:
        return

    w, h = frames[0].size
    canvas = Image.new("RGB", (w * len(frames), h))

    for i, frame in enumerate(frames):
        canvas.paste(frame, (i * w, 0))

    canvas.save(out_path)