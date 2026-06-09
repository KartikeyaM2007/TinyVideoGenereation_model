from pathlib import Path
import csv
import argparse

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data/videos/RoadAccidents")
    parser.add_argument("--out", type=str, default="data/captions.csv")
    parser.add_argument("--prompt", type=str, default="road accident surveillance scene")
    args = parser.parse_args()

    folder = Path(args.folder)
    videos = sorted([p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS])

    if not videos:
        raise FileNotFoundError(f"No videos found in {folder}")

    video = videos[0]
    print(f"Using video: {video}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "prompt"])
        writer.writeheader()
        writer.writerow({
            "video_path": str(video).replace("\\", "/"),
            "prompt": args.prompt
        })

    print(f"Saved one row to {out}")

if __name__ == "__main__":
    main()