import argparse
import os
import cv2
import yaml
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm


CLASS_TO_ID = {
    "Normal": 0,
    "Fighting": 1,
    "RoadAccidents": 2,
}


def read_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_indices(total_frames, n_samples):
    if total_frames <= 0:
        return []
    if total_frames <= n_samples:
        return list(range(total_frames))
    step = total_frames / n_samples
    return [int(i * step) for i in range(n_samples)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/image2d_class.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    data_cfg = cfg["data"]

    videos_root = Path(data_cfg["videos_root"])
    output_dir = Path(data_cfg["output_dir"])
    metadata_csv = Path(data_cfg["metadata_csv"])
    classes = data_cfg["classes"]
    resolution = int(data_cfg["resolution"])
    frames_per_video = int(data_cfg["frames_per_video"])
    max_videos_per_class = int(data_cfg["max_videos_per_class"])

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for class_name in classes:
        class_dir = videos_root / class_name
        if not class_dir.exists():
            print(f"[skip] missing class folder: {class_dir}")
            continue

        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

        videos = sorted([
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ])
        random.shuffle(videos)

        if max_videos_per_class > 0:
            videos = videos[:max_videos_per_class]

        class_out = output_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        for video_path in tqdm(videos, desc=f"Extracting {class_name}"):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idxs = set(sample_indices(total_frames, frames_per_video))

            frame_id = 0
            saved = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id in idxs:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (resolution, resolution))
                    out_name = f"{video_path.stem}_frame_{frame_id:05d}.png"
                    out_path = class_out / out_name
                    cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    rows.append({
                        "image_path": str(out_path).replace("\\", "/"),
                        "label_id": CLASS_TO_ID[class_name],
                        "label_name": class_name,
                    })
                    saved += 1

                frame_id += 1

            cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(metadata_csv, index=False)
    print(f"Saved {len(df)} frame rows to {metadata_csv}")


if __name__ == "__main__":
    main()