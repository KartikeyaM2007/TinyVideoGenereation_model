# AnomVidGen Hugging Face Space Packaging Fix

Use this Python-based packager instead of the broken PowerShell-only script.
It avoids PowerShell quote and here-string parsing issues.

## Files

- `prepare_hf_space.py`
- `prepare_hf_space_python.ps1`

## Install location

Copy both files into your project root:

```text
E:\TinyVideoDiffusinModel
```

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_hf_space_python.ps1
```

or directly:

```powershell
python prepare_hf_space.py
```

## Test prepared Space locally

```powershell
cd hf_space
powershell -ExecutionPolicy Bypass -File .\test_space_locally.ps1
```

## Push to Hugging Face Spaces

Inside `hf_space`:

```powershell
git init
git lfs install
git add .
git commit -m "Initial AnomVidGen Space"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```
