You are working inside a project named TinyVideoDiffusinModel.

Goal: build a low VRAM text to video diffusion MVP for anomaly videos on RTX 3050 4GB VRAM.

Architecture:
- Frozen MiniLM text encoder for prompt embeddings.
- Train a tiny 3D U-Net diffusion denoiser from scratch.
- Dataset contains short anomaly videos paired with simple prompts.
- Output should be 64x64, 8 frame GIFs first.

Hard constraints:
- Optimize for 4GB VRAM.
- Use batch size 1.
- Use fp16 mixed precision if CUDA is available.
- Avoid Transformers for video model.
- Do not implement large attention blocks first.
- Use FiLM conditioning from text embeddings.
- Keep code simple and modular.

Build these files:
- requirements.txt
- configs/default.yaml
- data/captions.csv example
- scripts_extract_clips.py
- scripts_precompute_text.py
- train.py
- sample.py
- app.py
- src/data/video_utils.py
- src/data/dataset.py
- src/text/ optional helpers
- src/models/unet3d.py
- src/diffusion/gaussian.py
- src/utils/config.py

Pipeline commands:
1. python scripts_extract_clips.py --config configs/default.yaml
2. python scripts_precompute_text.py --config configs/default.yaml
3. python train.py --config configs/default.yaml
4. python sample.py --config configs/default.yaml --prompt "a person falling" --output outputs/fall.gif
5. streamlit run app.py

Implementation notes:
- captions.csv columns: video_path,prompt
- extracted clip format: compressed npz with key frames, shape T,H,W,C, uint8
- dataset tensor shape: C,T,H,W normalized to [-1,1]
- model input shape: B,C,T,H,W
- diffusion loss: predict Gaussian noise with MSE
- sampling: start from noise and denoise into video
- classifier free guidance: randomly zero text embeddings during training

After building, verify:
- python -m compileall .
- scripts can import project modules
- train.py prints CUDA device when available
