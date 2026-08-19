# System Architecture

## Inference Pipeline

```
                 User
                  │
                  ▼
           Gradio / API
                  │
                  ▼
         Image Preprocessing
          (RGB, 768×768)
                  │
                  ▼
            SDXL Img2Img
                  +
            LoRA-500 Adapter
                  │
                  ▼
          Comic Image Output
```

## Detailed Pipeline

```
Input Image
     ↓
RGB Conversion
     ↓
768×768 Resize
     ↓
Latent Encoding (VAE Encoder)
     ↓
Noise Addition (strength=0.45)
     ↓
Diffusion Denoising (20 steps)
     + LoRA Adaptation
     + Text Conditioning (prompt)
     + Classifier-Free Guidance (scale=7)
     ↓
VAE Decode
     ↓
Comic Output (768×768 RGB)
```

## Model Architecture

### Base Model: SDXL 1.0

Stable Diffusion XL is a latent diffusion model with:

- **UNet**: 2.6B parameter denoising network
- **Text Encoders**: CLIP ViT-L + OpenCLIP ViT-bigG (dual text conditioning)
- **VAE**: Variational autoencoder for image ↔ latent space conversion
- **Scheduler**: Controls the noise schedule during denoising

### LoRA Adapter

Low-Rank Adaptation adds trainable matrices to the UNet attention layers:

```
Original weight: W (frozen)
LoRA matrices:   A (rank r) × B (rank r)
Effective:       W + A × B
```

- **Adapter size**: ~23 MB (vs 6.9 GB base model)
- **Rank**: Default (controlled by Diffusers)
- **Target modules**: UNet attention layers

### Img2Img Pipeline

Unlike text-to-image, Img2Img:

1. Encodes the input image to latent space
2. Adds noise proportional to `strength` (0.45 = 45% noise)
3. Denoises from the partially noised latent
4. Preserves structural information from the input

This is critical for **identity preservation** — the model translates style while retaining facial structure.

## Training Architecture

```
480 Training Pairs
     ↓
DataLoader (batch=1/GPU)
     ↓
┌─────────┬─────────┐
│  GPU 0  │  GPU 1  │    NCCL Backend
│ (rank 0)│ (rank 1)│
└────┬────┴────┬────┘
     │         │
     ▼         ▼
Gradient Accumulation (×4)
     │
     ▼
8-bit Adam Optimizer
     │
     ▼
LoRA Weight Update
     │
     ▼
Checkpoint (every 250 steps)
     │
     ▼
Final Adapter (step 500)
```

## Code Architecture

```
comic-image-translation/
│
├── inference/pipeline.py     ← ComicPipeline class
│       │
│       ├── Used by app.py         (Gradio)
│       ├── Used by scripts/       (CLI)
│       └── Used by evaluation/    (metrics)
│
├── evaluation/metrics.py     ← ImageMetrics + CLIPSimilarity
│       │
│       └── Used by evaluation/evaluate.py
│
├── training/train_lora.py    ← Reproducible training
│
└── configs/                  ← All parameters centralized
        ├── config.yaml            (inference)
        └── training_config.yaml   (training)
```
