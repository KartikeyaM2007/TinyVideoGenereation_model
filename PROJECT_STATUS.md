# Project Status

## Final stable route

Clean CCTV/anomaly videos
→ class conditioned image diffusion
→ class conditioned 2 frame diffusion
→ Streamlit demo

## Final supported classes

- Normal
- Fighting
- RoadAccidents

## Final files

- app_final.py
- configs/image2d_class.yaml
- configs/video2f_class.yaml
- train_image_diffusion.py
- train_2frame_diffusion.py
- sample_image_diffusion.py
- sample_2frame_diffusion.py
- src/image2d/
- src/video2f/

## Experiments

The project also contains earlier experiments for text-conditioned generation, 4-frame generation, and transition-based generation.

## Final decision

The 2-frame diffusion model gave the best quality and stability on RTX 3050 4GB VRAM.
