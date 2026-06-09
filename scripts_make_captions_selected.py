import argparse
import csv
import random
from pathlib import Path

PROMPT_MAP = {
    "Normal": "normal surveillance scene",
    "Normal1": "normal surveillance scene",
    "Fighting": "fighting surveillance scene",
    "RoadAccidents": "road accident surveillance scene",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/videos")
    parser.add_argument("--out", type=str, default="data/captions.csv")
    parser.add_argument("--classes", nargs="+", default=["Normal", "Fighting", "RoadAccidents"])
    parser.add_argument("--max-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    rows = []

    for class_name in args.classes:
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"[skip] missing folder: {class_dir}")
            continue

        prompt = PROMPT_MAP.get(class_name, class_name.lower().replace("_", " "))
        videos = [
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in VIDEO_EXTS
        ]

        random.shuffle(videos)

        if args.max_per_class > 0:
            videos = videos[:args.max_per_class]

        for video_path in videos:
            rows.append({
                "video_path": str(video_path).replace("\\", "/"),
                "prompt": prompt,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")

    counts = {}
    for row in rows:
        class_name = Path(row["video_path"]).parent.name
        counts[class_name] = counts.get(class_name, 0) + 1

    print("\nClass counts:")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()