# Evaluation

## Metrics

The model is evaluated using three complementary image quality metrics.

### LPIPS (Learned Perceptual Image Patch Similarity)

Measures perceptual similarity using deep features from AlexNet.

- **Lower values** indicate greater perceptual similarity.
- Range: 0.0 (identical) to 1.0+ (very different).
- More aligned with human perception than pixel-level metrics.

### SSIM (Structural Similarity Index)

Measures structural similarity based on luminance, contrast, and structure.

- **Higher values** indicate greater structural similarity.
- Range: -1.0 to 1.0 (1.0 = identical).
- Captures local structural patterns rather than global features.

### CLIP Similarity

Cosine similarity between CLIP ViT-B/32 image embeddings.

- **Higher values** indicate greater semantic/image-domain similarity.
- Range: -1.0 to 1.0.
- Captures high-level semantic alignment between generated and target images.

## Final Results

| Model | LPIPS ↓ | SSIM ↑ | CLIP ↑ |
|---|---:|---:|---:|
| Base SDXL | 0.639657 | 0.404214 | 0.649730 |
| **SDXL + LoRA-500 (2 GPU)** | **0.562685** | **0.414692** | **0.821407** |

### Improvement over Baseline

| Metric | Improvement |
|---|---|
| LPIPS | 12.03% lower (better) |
| SSIM | 2.59% higher (better) |
| CLIP | 26.43% higher (better) |

## Metrics Not Reported

### FID (Fréchet Inception Distance)

FID was planned but was not reliably executed in the Kaggle environment because of a Torch-Fidelity dependency incompatibility. No FID score is reported rather than reporting an unreliable number.

### Identity Preservation

Automated face recognition (for measuring identity consistency between input and output) showed limited detection coverage on stylized comic outputs. Comic-style faces often fail standard face detection models.

Identity preservation is therefore treated as a supplementary qualitative evaluation rather than a quantitative metric.

## Running Evaluation

```bash
python -m evaluation.evaluate \
    --metadata docs/metadata.csv \
    --generated-dir outputs/final_test_lora500_2gpu \
    --output results/validation_results.csv
```

**Note:** The metadata CSV paths may need adjustment if running outside the original Kaggle environment. Update paths to point to local dataset locations.
