import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


KAGGLE_DATASETS = {
    # Fighting / Violence
    "scvd": {
        "type": "dataset",
        "slug": "toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd",
        "out": "data/downloads/scvd",
        "class_hint": "Fighting",
    },
    "cctv_fights": {
        "type": "dataset",
        "slug": "shreyj1729/cctv-fights-dataset",
        "out": "data/downloads/cctv_fights",
        "class_hint": "Fighting",
    },

    # Road accident / traffic incident
    "road_accidents_cctv": {
        "type": "dataset",
        "slug": "suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset",
        "out": "data/downloads/road_accidents_cctv",
        "class_hint": "RoadAccidents",
    },
    "hwid12": {
        "type": "dataset",
        "slug": "landrykezebou/hwid12-highway-incidents-detection-dataset",
        "out": "data/downloads/hwid12",
        "class_hint": "RoadAccidents",
    },

    # Accident frames/images, optional but useful for image diffusion
    "accident_detection_frames": {
        "type": "dataset",
        "slug": "ckay16/accident-detection-from-cctv-footage",
        "out": "data/downloads/accident_detection_frames",
        "class_hint": "RoadAccidents",
    },

    # Kaggle competition, may require accepting rules first
    "accident_cvpr": {
        "type": "competition",
        "slug": "accident",
        "out": "data/downloads/accident_cvpr",
        "class_hint": "RoadAccidents",
    },
}


OFFICIAL_PAGES = {
    "NTU CCTV-Fights official page": "https://rose1.ntu.edu.sg/dataset/cctvFights/",
    "ACCIDENT benchmark official page": "https://accidentbench.github.io/",
    "CADP traffic accident project page": "https://ankitshah009.github.io/accident_forecasting_traffic_camera",
    "CADP OpenDataLab download page": "https://opendatalab.com/OpenDataLab/CADP/download",
}


def run(cmd: list[str], check: bool = True):
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def ensure_kaggle_installed():
    if shutil.which("kaggle") is not None:
        print("[ok] kaggle CLI found")
        return

    print("[info] kaggle CLI not found. Installing kaggle...")
    run([sys.executable, "-m", "pip", "install", "kaggle"])


def check_kaggle_token():
    home = Path.home()
    token_path = home / ".kaggle" / "kaggle.json"

    if token_path.exists():
        print(f"[ok] Kaggle token found: {token_path}")
        return True

    print("\n[missing] Kaggle API token not found.")
    print("Create/download kaggle.json from:")
    print("https://www.kaggle.com/settings/account")
    print("\nThen put it here:")
    print(token_path)
    print("\nOn Windows PowerShell:")
    print(r'mkdir $env:USERPROFILE\.kaggle -ErrorAction SilentlyContinue')
    print(r'copy kaggle.json $env:USERPROFILE\.kaggle\kaggle.json')
    return False


def download_kaggle_item(name: str, item: dict, force: bool = False):
    out_dir = Path(item["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / ".download_complete"

    if marker.exists() and not force:
        print(f"[skip] {name} already downloaded: {out_dir}")
        return

    if item["type"] == "dataset":
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            item["slug"],
            "-p",
            str(out_dir),
            "--unzip",
        ]
    elif item["type"] == "competition":
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            item["slug"],
            "-p",
            str(out_dir),
        ]
    else:
        raise ValueError(f"Unknown item type: {item['type']}")

    print(f"\n=== Downloading {name} ===")
    print(f"Class hint: {item['class_hint']}")
    print(f"Output: {out_dir}")

    code = run(cmd, check=False)

    if code == 0:
        marker.write_text("ok", encoding="utf-8")
        print(f"[done] {name}")
    else:
        print(f"[failed] {name}")
        if item["type"] == "competition":
            print("This may require accepting competition rules on Kaggle first.")
        print("Continue with the next dataset...")


def make_clean_dirs():
    base = Path("data/videos_clean")
    for sub in ["Normal", "Fighting", "RoadAccidents"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

    print("\n[ok] Created clean target folders:")
    print(base / "Normal")
    print(base / "Fighting")
    print(base / "RoadAccidents")


def print_tree_hint():
    print("\nDownloaded datasets are inside:")
    print("data/downloads/")
    print("\nClean training folders are inside:")
    print("data/videos_clean/")
    print("\nNext step after download:")
    print("Copy useful .mp4 clips into:")
    print("data/videos_clean/Fighting/")
    print("data/videos_clean/RoadAccidents/")
    print("data/videos_clean/Normal/")
    print("\nThen change your config:")
    print("videos_root: data/videos_clean")


def print_official_pages():
    print("\nOfficial/manual pages:")
    for name, url in OFFICIAL_PAGES.items():
        print(f"- {name}: {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=f"Download only selected names. Options: {list(KAGGLE_DATASETS.keys())}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if marker exists.",
    )
    parser.add_argument(
        "--include-competition",
        action="store_true",
        help="Also try Kaggle competition download for ACCIDENT. Requires rules accepted.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List datasets and exit.",
    )
    args = parser.parse_args()

    print("\nAvailable Kaggle items:")
    for name, item in KAGGLE_DATASETS.items():
        print(f"- {name}: {item['slug']} -> {item['out']}")

    print_official_pages()

    if args.list:
        return

    make_clean_dirs()
    ensure_kaggle_installed()

    if not check_kaggle_token():
        print("\nFix kaggle.json first, then rerun:")
        print("python download_datasets.py")
        return

    selected = args.only or list(KAGGLE_DATASETS.keys())

    if not args.include_competition and "accident_cvpr" in selected:
        selected.remove("accident_cvpr")

    for name in selected:
        if name not in KAGGLE_DATASETS:
            print(f"[skip] unknown dataset name: {name}")
            continue

        download_kaggle_item(name, KAGGLE_DATASETS[name], force=args.force)

    print_tree_hint()


if __name__ == "__main__":
    main()