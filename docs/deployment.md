# Production Deployment Guide

## Overview

This document describes the production deployment architecture for the comic style transfer service.

## Deployment Options

### Option 1: Hugging Face Spaces (Recommended for Demo)

Deploy the Gradio application directly to Hugging Face Spaces:

```bash
# 1. Create a new Space on huggingface.co
# 2. Push the repository
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/comic-image-translation
git push space main
```

**Requirements:**
- GPU Space (T4 or A10G)
- ~8 GB VRAM minimum

### Option 2: Docker Container

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Download model (or mount as volume)
RUN python scripts/download_model.py --repo YOUR_REPO

EXPOSE 7860
CMD ["python", "app.py"]
```

### Option 3: Cloud GPU Instance

Deploy on AWS (p3.2xlarge), GCP (n1-standard-4 + T4), or Azure (NC4as_T4_v3):

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download model
python scripts/download_model.py --repo YOUR_REPO

# Launch
python app.py --server-name 0.0.0.0 --server-port 7860
```

## Infrastructure Requirements

### GPU Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 6 GB | 8+ GB |
| GPU Type | T4 | A10G / A100 |
| System RAM | 16 GB | 32 GB |
| Storage | 10 GB | 20 GB |

### Latency

| Metric | Value |
|---|---|
| Inference latency | ~14 seconds/image |
| Model loading | ~30 seconds (first request) |
| Peak VRAM | 5.81 GB allocated |

## Scaling

### Horizontal Scaling

For multiple concurrent users:

```
Load Balancer
     │
     ├── GPU Instance 1 (Gradio)
     ├── GPU Instance 2 (Gradio)
     └── GPU Instance N (Gradio)
```

Each instance handles one request at a time due to GPU memory constraints.

### Batch Inference

For bulk processing:

```python
# Process multiple images sequentially
for image_path in image_paths:
    result = pipeline.generate(image=load(image_path), ...)
    result.save(output_path)
```

True batching is limited by VRAM — batch size 1 is recommended at 768×768.

## Optimization Opportunities

### Model Optimization

| Technique | Impact | Status |
|---|---|---|
| FP16 inference | ~50% VRAM reduction | ✅ Implemented |
| CPU offloading | Fits larger models | ✅ Implemented |
| VAE slicing/tiling | Reduces peak VRAM | ✅ Implemented |
| LoRA merging | Eliminates adapter overhead | 📋 Future |
| INT8 quantization | Further VRAM reduction | 📋 Future |
| TensorRT compilation | 2-4× speedup | 📋 Future |
| ONNX export | Cross-platform deployment | 📋 Future |

### Caching

- **Model caching**: Load model once, serve multiple requests
- **Prompt encoding**: Cache text encoder outputs (same prompt every time)
- **Latent caching**: Not applicable (unique inputs)

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|---|---|---|
| Inference latency | <15s | >30s |
| GPU utilization | >80% during inference | <10% idle |
| VRAM usage | <6 GB | >7.5 GB |
| Error rate | <1% | >5% |
| Queue depth | <5 | >20 |

### Health Check

```python
# GET /health
{
    "status": "healthy",
    "gpu_available": true,
    "model_loaded": true,
    "vram_used_gb": 5.81,
    "vram_total_gb": 16.0
}
```

### Logging

Log every inference request:

```json
{
    "timestamp": "2026-08-19T10:00:00Z",
    "latency_seconds": 13.85,
    "strength": 0.45,
    "guidance_scale": 7.0,
    "steps": 20,
    "resolution": 768,
    "success": true
}
```

## Cost Estimation

### Per-Image Cost

```
Cost/image = GPU hourly rate × (latency_seconds / 3600)
```

| GPU Type | Hourly Rate | Cost/Image |
|---|---:|---:|
| T4 (spot) | $0.35 | $0.0013 |
| T4 (on-demand) | $1.00 | $0.0038 |
| A10G | $1.50 | $0.0058 |
| A100 | $3.00 | $0.0115 |

### Monthly Estimates (1000 images/day)

| GPU Type | Monthly Cost |
|---|---:|
| T4 (spot) | ~$40 |
| T4 (on-demand) | ~$115 |
| A10G | ~$175 |

*Estimates exclude storage, networking, and compute for non-GPU tasks.*

## Security

- Input validation: Check image format, size, dimensions
- Rate limiting: Prevent abuse
- File size limits: Max 10 MB per upload
- No PII storage: Process and discard
- HTTPS: Required for production
