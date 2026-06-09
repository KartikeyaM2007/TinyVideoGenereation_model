import argparse
import csv
import random
from pathlib import Path

PROMPT_MAP = {
    "Abuse": "a person being abused in surveillance footage",
    "Arrest": "a person getting arrested",
    "Arson": "fire or arson in surveillance footage",
    "Assault": "a physical assault incident",
    "Burglary": "a burglary incident",
    "Explosion": "an explosion incident",
    "Fighting": "two people fighting",
    "Normal": "normal surveillance footage",
    "Normal1": "normal surveillance footage",
    "RoadAccidents": "a road accident",
    "Robbery": "a robbery incident",
    "Shooting": "a shooting incident",
    "Shoplifting": "a shoplifting incident",
    "Stealing": "a stealing incident",
    "Vandalism": "a vandalism incident",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/videos")
    parser.add_argument("--out", type=str, default="data/captions.csv")
    parser.add_argument("--max-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        prompt = PROMPT_MAP.get(class_name, f"a {class_name.lower()} incident")

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

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")

    class_counts = {}
    for row in rows:
        class_name = Path(row["video_path"]).parent.name
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nClass counts:")
    for k, v in sorted(class_counts.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()