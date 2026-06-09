# AnomVidGen deployment scripts

Extract these files into your project root:

```text
E:\TinyVideoDiffusinModel
```

## 1. Run final app locally

```powershell
powershell -ExecutionPolicy Bypass -File .\run_final_app.ps1
```

This creates `app_final.py`, checks the required model checkpoints, and launches Streamlit.

## 2. Prepare Hugging Face Space folder

```powershell
powershell -ExecutionPolicy Bypass -File .\prepare_hf_space.ps1
```

This creates:

```text
hf_space/
├── app.py
├── requirements.txt
├── README.md
├── configs/
├── src/
├── sample_image_diffusion.py
├── sample_2frame_diffusion.py
└── runs/
```

## 3. Test Space locally

```powershell
cd hf_space
powershell -ExecutionPolicy Bypass -File .\test_space_locally.ps1
```

## 4. Push to Hugging Face

Create a new Hugging Face Space with Streamlit SDK, then push the contents of `hf_space/`.

Use Git LFS for `.pt` checkpoint files:

```powershell
git lfs install
git add .
git commit -m "Initial AnomVidGen Space"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```
