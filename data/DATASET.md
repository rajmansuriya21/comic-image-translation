# Dataset Documentation

## Source

**Comic Faces (paired, synthetic) v2** — Face2Comics v2.0.0 by Sxela

A paired image translation dataset containing realistic face photographs and their corresponding comic-style renderings.

## Task

Realistic face → Comic face (image-to-image domain adaptation)

## Statistics

| Split | Pairs |
|---|---:|
| Train | 480 |
| Validation | 60 |
| Test | 60 |
| **Total** | **600** |

**Seed:** 42

## Data Preparation

### Pair Matching

- Matched input (face) and target (comic) files by identical filenames
- Verified one-to-one correspondence between inputs and targets
- Removed any unmatched files from the selection

### Preprocessing

- Converted all images to RGB color space
- Standardized output format to PNG
- Maintained original aspect ratios during initial processing

### Augmentation (Training)

- Random horizontal flip
- Center crop
- Resize to 768 × 768 pixels

### Augmentation (Validation / Test)

- Center crop
- Resize to 768 × 768 pixels
- No random augmentation (deterministic evaluation)

## Dataset Limitation

> **Important:** The original assignment requested paired Marvel movie-to-comic images. An approved paired realistic-face-to-comic dataset was used as a proxy because an appropriate paired Marvel dataset was not available. Therefore, the current model demonstrates realistic-face-to-comic domain adaptation rather than Marvel-specific character conversion.

## Class Distribution

The proxy dataset is a paired image translation dataset rather than a class-based classification dataset. Conventional class distribution metrics are therefore not applicable.

The dataset contains diverse:

- Facial expressions (neutral, smiling, serious)
- Lighting conditions (studio, natural, mixed)
- Accessories (glasses, hats, jewelry)
- Backgrounds (solid, gradient, natural)
- Demographics (varied age, gender, ethnicity)

## Storage

The full dataset is **not included** in this repository due to size constraints.

To reproduce:

1. Download the Face2Comics v2.0.0 dataset
2. Run the dataset preparation pipeline from the experiment notebook
3. The resulting splits will match the metadata in `docs/metadata.csv`
