# Training

## Final Experiment

The final model was trained using SDXL 1.0 as the base model with LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

### Configuration

| Parameter | Value |
|---|---|
| Base model | SDXL 1.0 |
| Training examples | 480 |
| Validation examples | 60 |
| Test examples | 60 |
| Resolution | 768 × 768 |
| Batch size per GPU | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 8 (2 GPUs × 1 × 4) |
| GPUs | 2 |
| Distributed backend | NCCL |
| Mixed precision | FP16 |
| Learning rate | 1e-4 |
| Scheduler | Constant |
| Optimizer | 8-bit Adam |
| Training steps | 500 |
| Checkpointing | Every 250 steps |
| Seed | 42 |

### Final Checkpoint

The selected final checkpoint is **LoRA-500**.

The trained adapter is stored as `comic_sdxl_lora_500.safetensors`.

## Design Decisions

### Why LoRA?

Full fine-tuning of SDXL (3.5B parameters) is impractical on consumer or cloud GPUs. LoRA adds low-rank adapter matrices to the attention layers, reducing trainable parameters to ~23 MB while preserving the base model's generalization capability.

### Why 768?

SDXL was trained on multiple resolutions including 768 × 768. Using 768 balances quality with memory usage — 1024 × 1024 would require significantly more VRAM for marginal quality improvement on face images.

### Why Batch Size 1?

SDXL Img2Img with LoRA at 768 resolution requires substantial VRAM. Batch size 1 per device avoids OOM errors while gradient accumulation of 4 across 2 GPUs achieves an effective batch size of 8.

### Why Gradient Accumulation?

Compensates for the small per-device batch size. With 4 accumulation steps across 2 GPUs, each optimization step sees 8 images, providing stable gradient estimates.

### Why 500 Steps?

Empirically determined. The LoRA adapter converges within 500 steps on this dataset size. More steps risk overfitting on 480 training examples.

### Why FP16?

Reduces memory usage by ~50% compared to FP32 with negligible quality loss. Required for fitting the SDXL pipeline on available GPUs.

### Why 2 GPUs?

Distributed training with 2 GPUs doubles throughput and allows the effective batch size of 8 without increasing per-device memory requirements.

### Why 8-bit Adam?

Further reduces memory usage by quantizing the optimizer states. Combined with FP16 and gradient checkpointing, this makes SDXL LoRA training feasible on GPUs with 16 GB VRAM.

## Dataset Limitation

The assignment requested paired Marvel movie-to-comic images.

An approved paired realistic-face-to-comic proxy dataset was used because a suitable paired Marvel dataset was not available. The model therefore demonstrates realistic-face-to-comic domain adaptation rather than Marvel-specific character conversion.

## Reproduction

```bash
python training/train_lora.py
```

This script will:

1. Clone the pinned Diffusers v0.37.1 repository
2. Launch distributed training with Accelerate
3. Save checkpoints at steps 250 and 500
4. Output the final LoRA adapter to `models/training_output/`

**Prerequisites:**

- Dataset prepared in `data/lora_train/` (Diffusers ImageFolder format)
- 2 GPUs with ≥16 GB VRAM each
- PyTorch with CUDA and NCCL support
