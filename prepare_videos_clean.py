import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}

FIGHTING_KEYWORDS = [
    "fight", "fighting", "violence", "violent", "abuse", "assault",
    "attack", "weapon", "robbery", "stealing", "shoplifting"
]

ROAD_KEYWORDS = [
    "accident", "crash", "collision", "road", "traffic", "vehicle",
    "car", "highway", "incident", "roadaccidents", "road_accidents"
]

NORMAL_KEYWORDS = [
    "normal", "nonviolence", "non-violence", "non_violence",
    "no violence", "noviolence", "negative", "safe", "nonaccident",
    "non-accident", "no_accident", "no accident", "normal1"
]


def get_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frames <= 0:
        return -1.0

    return float(frames / fps)


def classify_video(path: Path):
    text = str(path).lower().replace("\\", "/")

    # Important: normal first, so non-violence does not become violence.
    if any(k in text for k in NORMAL_KEYWORDS):
        return "Normal"

    if any(k in text for k in FIGHTING_KEYWORDS):
        return "Fighting"

    if any(k in text for k in ROAD_KEYWORDS):
        return "RoadAccidents"

    return None


def stable_name(path: Path, class_name: str):
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    clean_stem = path.stem.replace(" ", "_").replace("-", "_")
    return f"{class_name}_{clean_stem}_{h}{path.suffix.lower()}"


def scan_videos(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def source_key(path: Path, downloads_root: Path):
    rel = path.relative_to(downloads_root)
    parts = rel.parts

    if len(parts) >= 3:
        return str(Path(parts[0]) / parts[1] / parts[2])

    if len(parts) >= 2:
        return str(Path(parts[0]) / parts[1])

    return parts[0]


def balanced_select(items, downloads_root: Path, max_total: int, max_per_source: int, seed: int):
    random.seed(seed)

    grouped = defaultdict(list)

    for video_path, duration in items:
        key = source_key(video_path, downloads_root)
        grouped[key].append((video_path, duration))

    for key in grouped:
        random.shuffle(grouped[key])
        grouped[key] = grouped[key][:max_per_source]

    selected = []
    keys = list(grouped.keys())
    random.shuffle(keys)

    while len(selected) < max_total:
        added = False

        for key in keys:
            if grouped[key]:
                selected.append(grouped[key].pop())
                added = True

                if len(selected) >= max_total:
                    break

        if not added:
            break

    random.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads-root", type=str, default="data/downloads")
    parser.add_argument("--out-root", type=str, default="data/videos_clean")
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--max-per-source", type=int, default=40)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    downloads_root = Path(args.downloads_root)
    out_root = Path(args.out_root)

    class_dirs = {
        "Normal": out_root / "Normal",
        "Fighting": out_root / "Fighting",
        "RoadAccidents": out_root / "RoadAccidents",
    }

    for d in class_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    videos = scan_videos(downloads_root)
    print(f"Found {len(videos)} video files under {downloads_root}")

    buckets = {
        "Normal": [],
        "Fighting": [],
        "RoadAccidents": [],
        "Unknown": [],
        "SkippedDuration": [],
    }

    for video in tqdm(videos, desc="Scanning videos"):
        class_name = classify_video(video)

        if class_name is None:
            buckets["Unknown"].append(video)
            continue

        duration = get_duration_seconds(video)

        if duration < args.min_duration or duration > args.max_duration:
            buckets["SkippedDuration"].append((video, duration, class_name))
            continue

        buckets[class_name].append((video, duration))

    print("\nDetected usable videos:")
    for class_name in ["Normal", "Fighting", "RoadAccidents"]:
        print(f"{class_name}: {len(buckets[class_name])}")

    print(f"Unknown: {len(buckets['Unknown'])}")
    print(f"Skipped by duration: {len(buckets['SkippedDuration'])}")

    selected_by_class = {}

    for class_name in ["Normal", "Fighting", "RoadAccidents"]:
        selected_by_class[class_name] = balanced_select(
            buckets[class_name],
            downloads_root=downloads_root,
            max_total=args.max_per_class,
            max_per_source=args.max_per_source,
            seed=args.seed,
        )

    print("\nSelected videos:")
    for class_name, items in selected_by_class.items():
        print(f"{class_name}: {len(items)}")

    if args.dry_run:
        print("\nDry run only. No files copied.")

        for class_name in ["Normal", "Fighting", "RoadAccidents"]:
            print(f"\n{class_name} examples:")
            for video_path, duration in selected_by_class[class_name][:12]:
                print(f"  {video_path}  duration={duration:.2f}s")

        return

    copied_counts = {
        "Normal": 0,
        "Fighting": 0,
        "RoadAccidents": 0,
    }

    for class_name, items in selected_by_class.items():
        for src_path, duration in tqdm(items, desc=f"Copying {class_name}"):
            dst_name = stable_name(src_path, class_name)
            dst_path = class_dirs[class_name] / dst_name

            if dst_path.exists() and not args.force:
                continue

            shutil.copy2(src_path, dst_path)
            copied_counts[class_name] += 1

    print("\nCopied videos:")
    for class_name, count in copied_counts.items():
        print(f"{class_name}: {count}")

    print("\nFinal clean dataset:")
    print(out_root)

    print("\nNext config change:")
    print("videos_root: data/videos_clean")


if __name__ == "__main__":
    main()