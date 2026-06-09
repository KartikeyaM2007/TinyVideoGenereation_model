from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path.cwd()
SPACE_DIR = ROOT / "hf_space"

REQUIRED = [
    "app_final.py",
    "sample_image_diffusion.py",
    "sample_2frame_diffusion.py",
    "configs/image2d_class.yaml",
    "configs/video2f_class.yaml",
    "src/image2d",
    "src/video2f",
    "runs/image2d_class/latest.pt",
    "runs/video2f_class/latest.pt",
]

REQUIREMENTS_TXT = """streamlit
torch
torchvision
pyyaml
pillow
numpy
pandas
opencv-python-headless
tqdm
"""

README_MD = """---
title: AnomVidGen
emoji: 🎥
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# AnomVidGen

Low VRAM anomaly image and 2 frame video generation using compact PyTorch diffusion models.

## Supported classes

- Normal
- Fighting
- RoadAccidents

## Final pipeline

```text
class/text input
        ↓
class-conditioned image diffusion
        ↓
class-conditioned 2 frame diffusion
        ↓
surveillance-style image or 2 frame GIF
```

## Notes

This is not a general text-to-video model. Text prompts are mapped to fixed anomaly classes.

The final stable version focuses on image generation and 2 frame GIF generation for a low VRAM setup.
"""

GITIGNORE = """__pycache__/
*.pyc
outputs/
.DS_Store
Thumbs.db
"""

GITATTRIBUTES = """*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
"""

TEST_SPACE_LOCALLY_PS1 = """$ErrorActionPreference = \"Stop\"
Set-Location $PSScriptRoot
python -m pip install -r requirements.txt
streamlit run app.py
"""


def fail(message: str) -> None:
    print(f"[error] {message}")
    raise SystemExit(1)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src_path = ROOT / src if isinstance(src, str) else src
    dst_path = SPACE_DIR / dst if isinstance(dst, str) else dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    print(f"[copy] {src_path.relative_to(ROOT)} -> {dst_path.relative_to(ROOT)}")


def copy_dir(src: str | Path, dst: str | Path) -> None:
    src_path = ROOT / src if isinstance(src, str) else src
    dst_path = SPACE_DIR / dst if isinstance(dst, str) else dst
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_path, dst_path)
    print(f"[copy-dir] {src_path.relative_to(ROOT)} -> {dst_path.relative_to(ROOT)}")


def write_text(path: str, content: str) -> None:
    out_path = SPACE_DIR / path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"[write] {out_path.relative_to(ROOT)}")


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    print("\n=== Preparing AnomVidGen Hugging Face Space with Python ===\n")
    print(f"Project root: {ROOT}")

    missing = []
    for item in REQUIRED:
        if not (ROOT / item).exists():
            missing.append(item)

    if missing:
        print("\nMissing required files/folders:")
        for item in missing:
            print(f"  - {item}")
        fail("Create/train missing files before packaging the Space.")

    if SPACE_DIR.exists():
        print(f"[remove] {SPACE_DIR.relative_to(ROOT)}")
        shutil.rmtree(SPACE_DIR)

    # Folders
    for folder in [
        "configs",
        "src",
        "runs/image2d_class",
        "runs/video2f_class",
        "outputs/app",
        "outputs/video2f_samples",
    ]:
        (SPACE_DIR / folder).mkdir(parents=True, exist_ok=True)
        print(f"[dir] hf_space/{folder}")

    # Core app and scripts
    copy_file("app_final.py", "app.py")
    copy_file("sample_image_diffusion.py", "sample_image_diffusion.py")
    copy_file("sample_2frame_diffusion.py", "sample_2frame_diffusion.py")

    # Configs
    copy_file("configs/image2d_class.yaml", "configs/image2d_class.yaml")
    copy_file("configs/video2f_class.yaml", "configs/video2f_class.yaml")

    # Source packages
    copy_dir("src/image2d", "src/image2d")
    copy_dir("src/video2f", "src/video2f")
    (SPACE_DIR / "src/__init__.py").touch()

    # Checkpoints
    copy_file("runs/image2d_class/latest.pt", "runs/image2d_class/latest.pt")
    copy_file("runs/video2f_class/latest.pt", "runs/video2f_class/latest.pt")

    # Metadata/support files
    write_text("requirements.txt", REQUIREMENTS_TXT)
    write_text("README.md", README_MD)
    write_text(".gitignore", GITIGNORE)
    write_text(".gitattributes", GITATTRIBUTES)
    write_text("test_space_locally.ps1", TEST_SPACE_LOCALLY_PS1)

    print("\nCheckpoint sizes:")
    for ckpt in [
        SPACE_DIR / "runs/image2d_class/latest.pt",
        SPACE_DIR / "runs/video2f_class/latest.pt",
    ]:
        print(f"  - {ckpt.relative_to(ROOT)}: {checkpoint_size_mb(ckpt):.2f} MB")

    print("\n=== Done ===")
    print("Prepared folder: hf_space")
    print("\nTest locally:")
    print("  cd hf_space")
    print("  powershell -ExecutionPolicy Bypass -File .\\test_space_locally.ps1")
    print("\nDeploy to Hugging Face Spaces:")
    print("  cd hf_space")
    print("  git init")
    print("  git lfs install")
    print("  git add .")
    print('  git commit -m "Initial AnomVidGen Space"')
    print("  git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME")
    print("  git push -u origin main")


if __name__ == "__main__":
    main()
