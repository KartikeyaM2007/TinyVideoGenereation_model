from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.video_utils import read_video_frames
from src.utils.config import load_config


def make_clips(frames, num_frames: int, stride: int, max_clips: int):
    needed = num_frames * stride
    if len(frames) < needed:
        return []
    max_start = len(frames) - needed
    if max_clips <= 1:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, num=max_clips, dtype=int).tolist()
    clips = []
    for s in starts:
        idxs = [s + i * stride for i in range(num_frames)]
        clip = np.stack([frames[i] for i in idxs], axis=0).astype(np.uint8)
        clips.append(clip)
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    videos_csv = Path(data_cfg["videos_csv"])
    clips_dir = Path(data_cfg["clips_dir"])
    metadata_csv = Path(data_cfg["metadata_csv"])
    clips_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(videos_csv)
    if not {"video_path", "prompt"}.issubset(df.columns):
        raise ValueError("captions.csv must contain columns: video_path,prompt")

    rows = []
    for video_idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting clips"):
        video_path = Path(row["video_path"])
        prompt = str(row["prompt"])
        try:
            frames = read_video_frames(video_path, resolution=int(data_cfg["resolution"]))
            clips = make_clips(
                frames,
                num_frames=int(data_cfg["num_frames"]),
                stride=int(data_cfg["frame_stride"]),
                max_clips=int(data_cfg["clips_per_video"]),
            )
        except Exception as e:
            print(f"[skip] {video_path}: {e}")
            continue

        for clip_idx, clip in enumerate(clips):
            out_path = clips_dir / f"clip_{video_idx:06d}_{clip_idx:03d}.npz"
            np.savez_compressed(out_path, frames=clip)
            rows.append({"clip_path": str(out_path), "prompt": prompt, "source_video": str(video_path)})

    pd.DataFrame(rows).to_csv(metadata_csv, index=False)
    print(f"Saved {len(rows)} clips to {clips_dir}")
    print(f"Saved metadata to {metadata_csv}")


if __name__ == "__main__":
    main()
