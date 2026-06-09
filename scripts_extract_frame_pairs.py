import argparse
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


def sample_start_indices(total_frames, pairs_per_video, frame_gap):
    max_start = total_frames - frame_gap - 1
    if max_start <= 1:
        return []

    if max_start <= pairs_per_video:
        return list(range(max_start))

    step = max_start / pairs_per_video
    return [int(i * step) for i in range(pairs_per_video)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/video2f_class.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    data_cfg = cfg["data"]

    videos_root = Path(data_cfg["videos_root"])
    output_dir = Path(data_cfg["output_dir"])
    metadata_csv = Path(data_cfg["metadata_csv"])
    classes = data_cfg["classes"]
    resolution = int(data_cfg["resolution"])
    pairs_per_video = int(data_cfg["pairs_per_video"])
    frame_gap = int(data_cfg["frame_gap"])
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
            start_indices = sample_start_indices(total_frames, pairs_per_video, frame_gap)

            if not start_indices:
                cap.release()
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            for start_idx in start_indices:
                idx1 = start_idx
                idx2 = start_idx + frame_gap

                if idx2 >= len(frames):
                    continue

                frame1 = cv2.cvtColor(frames[idx1], cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(frames[idx2], cv2.COLOR_BGR2RGB)

                frame1 = cv2.resize(frame1, (resolution, resolution))
                frame2 = cv2.resize(frame2, (resolution, resolution))

                out1 = class_out / f"{video_path.stem}_pair_{idx1:05d}_a.png"
                out2 = class_out / f"{video_path.stem}_pair_{idx1:05d}_b.png"

                cv2.imwrite(str(out1), cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(out2), cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR))

                rows.append({
                    "frame1_path": str(out1).replace("\\", "/"),
                    "frame2_path": str(out2).replace("\\", "/"),
                    "label_id": CLASS_TO_ID[class_name],
                    "label_name": class_name,
                })

    df = pd.DataFrame(rows)
    df.to_csv(metadata_csv, index=False)
    print(f"Saved {len(df)} frame pairs to {metadata_csv}")


if __name__ == "__main__":
    main()