# Complete Workflow and Model Details

## Project Workflow

AnomVidGen follows a low VRAM anomaly generation workflow. The project does not directly train a large text to video model. Instead, it breaks the problem into smaller and more stable stages that can run on an RTX 3050 4GB GPU.

```text
Raw CCTV/anomaly datasets
        ↓
Dataset cleaning and class filtering
        ↓
Clean 3 class video dataset
        ↓
Frame extraction for image generation
        ↓
Class conditioned image diffusion training
        ↓
Frame pair extraction for 2 frame generation
        ↓
Class conditioned 2 frame diffusion training
        ↓
Sampling scripts
        ↓
Streamlit demo
        ↓
GitHub + Hugging Face deployment
```

---

## Step 1: Dataset Collection

The project started with UCF Crime style anomaly videos, then moved to cleaner CCTV and traffic datasets because raw anomaly datasets often contain noisy labels, irrelevant footage, watermarks, title cards, and long normal segments.

Datasets tested or used include:

```text
UCF Crime style anomaly dataset
SCVD Smart City CCTV Violence Detection Dataset
NTU CCTV Fights Dataset
Road Accidents From CCTV Footages Dataset
HWID12 Highway Incidents Dataset
```

The final project focuses only on three stable classes:

```text
Normal
Fighting
RoadAccidents
```

---

## Step 2: Dataset Cleaning

The dataset is cleaned and reorganized into this structure:

```text
data/videos_clean/
├── Normal/
├── Fighting/
└── RoadAccidents/
```

The script used for this step is:

```text
prepare_videos_clean.py
```

This script scans downloaded datasets and copies usable videos into the final clean folder. It filters videos using:

```text
class name
source dataset
video duration
maximum videos per class
maximum videos per source
supported video extensions
```

Supported video extensions:

```text
.mp4
.avi
.mov
.mkv
.webm
.mpeg
.mpg
```

This cleaning step is important because generation models are more sensitive to noisy data than classification models.

---

## Step 3: Image Frame Extraction

For image generation, single frames are extracted from the clean video dataset.

Script used:

```text
scripts_extract_class_frames.py
```

Input:

```text
data/videos_clean/
```

Output:

```text
data/class_images/
```

Purpose:

```text
Create training images for class conditioned image diffusion.
```

Each extracted image keeps its class label:

```text
Normal = 0
Fighting = 1
RoadAccidents = 2
```

---

## Step 4: Image Diffusion Training

The first final model is a class conditioned image diffusion model.

Training script:

```text
train_image_diffusion.py
```

Config:

```text
configs/image2d_class.yaml
```

Checkpoint output:

```text
runs/image2d_class/latest.pt
```

The model learns to generate a single CCTV style image based on a selected class.

Example:

```text
class = RoadAccidents
        ↓
model generates a road accident style surveillance image
```

---

## Step 5: 2 Frame Pair Extraction

For short video generation, the project extracts two related frames from each video.

Script used:

```text
scripts_extract_frame_pairs.py
```

Input:

```text
data/videos_clean/
```

Output:

```text
data/frame_pairs/
```

Purpose:

```text
Create paired frame samples for 2 frame diffusion training.
```

Each training sample contains:

```text
frame 1
frame 2
class label
```

---

## Step 6: 2 Frame Diffusion Training

The second final model is a class conditioned 2 frame diffusion model.

Training script:

```text
train_2frame_diffusion.py
```

Config:

```text
configs/video2f_class.yaml
```

Checkpoint output:

```text
runs/video2f_class/latest.pt
```

Instead of using a heavy 3D video model, the project stacks two RGB frames together as channels.

```text
frame 1 = 3 channels
frame 2 = 3 channels
total input/output = 6 channels
```

So the model generates:

```text
[frame_1_RGB, frame_2_RGB]
```

This is converted into:

```text
2 frame GIF
side by side PNG
```

This approach is lightweight and stable on 4GB VRAM.

---

## Step 7: Sampling and Inference

Image generation script:

```text
sample_image_diffusion.py
```

Example command:

```powershell
python sample_image_diffusion.py --config configs/image2d_class.yaml --class_name RoadAccidents --output outputs/road.png --steps 100 --guidance 1.0
```

2 frame generation script:

```text
sample_2frame_diffusion.py
```

Example command:

```powershell
python sample_2frame_diffusion.py --config configs/video2f_class.yaml --class_name Fighting --steps 100 --guidance 1.0
```

Final app:

```text
app_final.py
```

Run:

```powershell
streamlit run app_final.py
```

---

# Models Used

## 1. Final Image Generation Model

Model name:

```text
UNet2DClassConditioned
```

Used in:

```text
src/image2d/unet2d.py
```

Diffusion class:

```text
GaussianDiffusion2D
```

Used in:

```text
src/image2d/diffusion2d.py
```

Purpose:

```text
Generate single CCTV style anomaly images.
```

Input:

```text
random noise
timestep
class ID
```

Output:

```text
generated 64x64 RGB image
```

Main model parameters:

| Parameter           | Value / Meaning                 |
| ------------------- | ------------------------------- |
| Model type          | 2D U Net                        |
| Conditioning        | Class conditioned               |
| Number of classes   | 3                               |
| Classes             | Normal, Fighting, RoadAccidents |
| Input channels      | 3                               |
| Output channels     | 3 noise prediction              |
| Resolution          | 64x64                           |
| Base channels       | 32                              |
| Channel multipliers | [1, 2, 4]                       |
| Diffusion type      | Gaussian diffusion              |
| Sampling type       | iterative denoising             |
| Checkpoint          | `runs/image2d_class/latest.pt`  |

Important config file:

```text
configs/image2d_class.yaml
```

---

## 2. Final 2 Frame Video Generation Model

Model name:

```text
UNet2DClassConditioned
```

Used in:

```text
src/image2d/unet2d.py
```

Diffusion class:

```text
GaussianDiffusion2D
```

Used in:

```text
src/image2d/diffusion2d.py
```

Purpose:

```text
Generate short 2 frame CCTV style anomaly GIFs.
```

Input:

```text
random noise
timestep
class ID
```

Output:

```text
two generated RGB frames
```

Frame representation:

```text
2 RGB frames stacked as 6 channels
```

Main model parameters:

| Parameter           | Value / Meaning                 |
| ------------------- | ------------------------------- |
| Model type          | 2D U Net                        |
| Conditioning        | Class conditioned               |
| Number of classes   | 3                               |
| Classes             | Normal, Fighting, RoadAccidents |
| Input channels      | 6                               |
| Output channels     | 6 noise prediction              |
| Frame count         | 2                               |
| Resolution          | 64x64                           |
| Base channels       | 32                              |
| Channel multipliers | [1, 2, 4]                       |
| Diffusion type      | Gaussian diffusion              |
| Sampling type       | iterative denoising             |
| Checkpoint          | `runs/video2f_class/latest.pt`  |

Important config file:

```text
configs/video2f_class.yaml
```

---

## 3. Earlier Text Conditioned Video Experiment

This was an early experiment, not the final model.

Model components:

```text
MiniLM text encoder
Tiny 3D U Net
Gaussian diffusion
```

Goal:

```text
text prompt → anomaly video
```

Result:

```text
mostly noise or unstable static outputs
```

Reason it was dropped:

```text
direct text to video training was too hard for 4GB VRAM and limited training data
```

Final status:

```text
kept as experiment
not used in final app
```

---

## 4. 4 Frame Diffusion Experiment

This was another experiment.

Representation:

```text
4 RGB frames stacked as 12 channels
```

Model:

```text
2D U Net with 12 input channels
```

Goal:

```text
generate 4 frame GIFs
```

Result:

```text
frames were structurally consistent but nearly identical
```

Problem:

```text
temporal collapse
```

Final status:

```text
kept as experiment
not used in final app
```

---

## 5. Transition Model Experiment

This experiment tried to predict the next frame from the current frame.

Input:

```text
current frame
class ID
```

Output:

```text
next frame
```

Goal:

```text
generate longer motion by repeatedly predicting the next frame
```

Result:

```text
motion appeared but quality drifted over time
```

Problem:

```text
autoregressive error accumulation
```

Final status:

```text
kept as experiment
not used in final app
```

---

# Important Technical Terms

## Diffusion Model

A diffusion model learns to generate data by reversing a noise process.

Training:

```text
clean image → add noise → model learns to predict noise
```

Sampling:

```text
random noise → remove noise step by step → generated image
```

---

## Gaussian Diffusion

Gaussian diffusion means the noise added during training follows a Gaussian distribution.

In this project, Gaussian diffusion is used for both:

```text
image generation
2 frame generation
```

---

## U Net

A U Net is a neural network architecture commonly used in image generation and segmentation.

It has:

```text
downsampling path
bottleneck
upsampling path
skip connections
```

In this project, the U Net predicts the noise that should be removed from the current noisy sample.

---

## Class Conditioning

Class conditioning means the model is given a class label during training and generation.

Example:

```text
class ID = 1
class name = Fighting
```

This tells the model what type of image or frame pair to generate.

---

## Timestep Embedding

Diffusion models denoise over many timesteps.

The timestep embedding tells the model how noisy the current sample is.

Early timestep:

```text
very noisy
```

Late timestep:

```text
almost clean
```

---

## Channel Stacking

Channel stacking means multiple frames are combined along the channel dimension.

For 2 frame generation:

```text
frame 1 RGB = 3 channels
frame 2 RGB = 3 channels
total = 6 channels
```

This allows a 2D model to generate multiple frames without using a heavy 3D video model.

---

## Temporal Collapse

Temporal collapse happens when a video model generates frames that look almost identical.

This happened in the 4 frame experiment.

The model learned visual consistency but not enough motion.

---

## Autoregressive Drift

Autoregressive drift happens when a model predicts the next frame repeatedly and small errors accumulate.

Example:

```text
frame 1 → frame 2 → frame 3 → frame 4
```

If frame 2 has an error, frame 3 is based on that error, and quality keeps degrading.

This happened in the transition model experiment.

---

## Checkpoint

A checkpoint is the saved model weights after training.

Final checkpoints:

```text
runs/image2d_class/latest.pt
runs/video2f_class/latest.pt
```

---

## Guidance Scale

Guidance scale controls how strongly the generation follows the class condition.

Recommended value:

```text
1.0
```

Higher values can increase class effect but may reduce image quality.

---

# Final Inference Workflow

## Image Generation

```text
User selects class
        ↓
App sends class name to sample_image_diffusion.py
        ↓
Class name is converted to class ID
        ↓
Random noise is created
        ↓
Image diffusion model denoises the sample
        ↓
Generated image is saved
        ↓
Streamlit displays the image
```

## 2 Frame GIF Generation

```text
User selects class
        ↓
App sends class name to sample_2frame_diffusion.py
        ↓
Class name is converted to class ID
        ↓
Random 6 channel noise is created
        ↓
2 frame diffusion model denoises the sample
        ↓
6 channels are split into two RGB frames
        ↓
Frames are saved as GIF and side by side PNG
        ↓
Streamlit displays the result
```

---

# Final Design Decision

The final project uses image diffusion and 2 frame diffusion because this was the best balance between quality, stability, and hardware limits.

```text
Direct text to video = unstable
4 frame generation = temporal collapse
transition model = drift
image diffusion = stable
2 frame diffusion = stable enough for final demo
```

Final architecture:

```text
Class conditioned 2D diffusion
        +
2 frame channel stacked generation
        +
Streamlit interface
```

This makes AnomVidGen a realistic low VRAM anomaly generation project instead of an overclaimed full text to video system.
