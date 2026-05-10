# Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model

<a href="https://arxiv.org/abs/2403.07764"><img src="https://img.shields.io/badge/arXiv-2403.07764-b31b1b.svg" height=22.5></a>
![teaser](assets/sm_teaser.jpg)
Our proposed framework, Stable-Makeup, is a novel diffusion-based method for makeup transfer that can robustly transfer a diverse range of real-world makeup styles, from light to extremely heavy makeup.

## Method Details
![method](https://github.com/Xiaojiu-z/Stable-Makeup/blob/main/assets/sm_method.jpg)
Given a source image $\mathit{I_s}$ , a reference makeup image $\mathit{I_m}$ and an obtained facial structure control image $\mathit{I_c}$ , Stable-Makeup utilizes D-P makeup encoder to encode $\mathit{I_m}$. Content and structural encoders are used to encode $\mathit{I_s}$ and $\mathit{I_c}$ respectively. With the aid of the makeup cross-attention layers, Stable-Makeup aligns the facial regions of $\mathit{I_s}$ and $\mathit{I_m}$ , enabling successful transfers the intricate makeup details. After content-structure decoupling training, Stable-Makeup further maintains content and structure of $\mathit{I_s}$ .

## Custom Extensions & Composable Makeup Transfer

In addition to the original paper implementation, this fork includes a **composable partial makeup transfer pipeline** that enables flexible, multi-reference makeup application:

### Pipeline Overview
```
prepare_id.py → generate_sd_makeup.py (×2 refs) → [cv_dlib.py | dl_alpha.py | dl_lab.py] → Composite Output
```

Before running the compositing scripts (`cv_dlib.py`, `dl_alpha.py`, or `dl_lab.py`), you must download the following weights and place them in the root directory of this project:

1. **[79999_iter.pth](https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth)** (53 MB)
   * *BiSeNet weights for semantic perception, trained on CelebAMask-HQ.*

2. **[shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)** (61 MB)
   * *Dlib weights for the classical CV baseline.*
   * **Note:** This file downloads as a compressed `.bz2` archive. You must extract it before use. On Linux/Mac, you can run: `bzip2 -d shape_predictor_68_face_landmarks.dat.bz2`

### Key Features

#### 1. **ID Dataset Preparation** (`prepare_id.py`)
- Samples and resizes 1000 images from FFHQ dataset to 512×512 resolution
- Prepares base ID images for pipeline processing

#### 2. **Makeup Generation** (`generate_sd_makeup.py`)
- Generates diverse makeup variants using Stable Diffusion v1.5 with LEDITS inversion
- Creates two reference makeup sets (A and B) from customizable prompts (`prompt.txt`)
- Uses DDIM sampling with semantic guidance for stable, controllable generation

#### 3. **Composable Partial Transfer** – Choose One of Three Methods

**Option A: Landmark-Based Transfer** (`cv_dlib.py`)
- **Method**: Dlib facial landmarks (68 points) for precise region identification
- **Features**: Eyes from Reference A (landmarks 36-47) + Lips from Reference B (landmarks 48-67)
- **Blending**: Poisson seamless cloning for natural boundaries
- **Speed**: Fast (~10-30 min for 1000 images)
- **Requirements**: `shape_predictor_68_face_landmarks.dat`

**Option B: Semantic Segmentation + Alpha Blending** (`dl_alpha.py`)
- **Method**: BiSeNet face parsing (19 classes) with soft alpha blending in RGB space
- **Features**: Eyes (classes 4,5) + Lips (classes 12,13) with skin-aware dilation
- **Blending**: Gaussian blur masks for smooth color transitions
- **Speed**: Moderate (~30-60 min for 1000 images with GPU)
- **Requirements**: BiSeNet weights (`79999_iter.pth`), CUDA GPU recommended

**Option C: LAB Color Space Transfer** (`dl_lab.py`)
- **Method**: BiSeNet parsing with advanced LAB color space blending
- **Features**: Full color transfer (A & B channels) + 50% lightness transfer (L channel)
- **Advantage**: Superior color fidelity and natural makeup appearance
- **Blending**: Decomposed color and lightness control for professional results
- **Speed**: Slowest (~60+ min for 1000 images with GPU)
- **Best For**: High-quality, realistic makeup transfer

### Output
All methods produce composited face images in `data/partial_transfer/` with combined eye makeup from Reference A and lip makeup from Reference B.

### Usage Example
```bash
# Step 1: Prepare dataset
python prepare_id.py

# Step 2: Generate two makeup references (modify prompts in prompt.txt for different styles)
python generate_sd_makeup.py
# Rename output folder to: data/makeup_image_A/
python generate_sd_makeup.py
# Rename output folder to: data/makeup_image_B/

# Step 3: Choose transfer method
python cv_dlib.py      # Fast landmark-based
# OR
python dl_alpha.py     # Semantic segmentation with RGB blending
# OR
python dl_lab.py       # Semantic segmentation with LAB blending
```

## Todo List
1. - [x] inference and training code
2. - [x] pre-trained weights

## Getting Started
### Environment Setup
Our code is built on the [diffusers](https://github.com/huggingface/diffusers/) version of Stable Diffusion v1-5. We use [SPIGA](https://github.com/andresprados/SPIGA) and [facelib](https://github.com/sajjjadayobi/FaceLib) to draw face structural images. 
```shell
git clone https://github.com/Xiaojiu-z/Stable-Makeup.git
cd Stable-Makeup
```
### Pretrained Models
[Google Drive](https://drive.google.com/drive/folders/1397t27GrUyLPnj17qVpKWGwg93EcaFfg?usp=sharing).
Download them and save them to the directory `models/stablemakeup`. One deviation from the original paper is randomly dropping out inputs into the structural encoder during training, resulting in improved semantic alignment. Enjoy it!

### Inference

```python
python infer_kps.py
```

### Training
You can prepare datasets following our paper and make a jsonl file (each line with 4 key-value pairs, including original id, edited id, augmented id, face structural image of edited id) or you can implement a dataset and a dataloader class by yourself (Probably faster than organizing into my data form).

```python
bash train.sh
```

### Gradio demo
We provide a simple gr demo for more flexible use.
```python
python gradio_demo_kps.py
```

## Citation
```
@article{zhang2024stable,
  title={Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model},
  author={Zhang, Yuxuan and Wei, Lifu and Zhang, Qing and Song, Yiren and Liu, Jiaming and Li, Huaxia and Tang, Xu and Hu, Yao and Zhao, Haibo},
  journal={arXiv preprint arXiv:2403.07764},
  year={2024}
}
```

