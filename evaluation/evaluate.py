"""
Evaluation script for comic style transfer models.

Evaluates a directory of generated images against target images
using LPIPS, SSIM, and CLIP similarity metrics.

Usage:
    python -m evaluation.evaluate \
        --metadata docs/metadata.csv \
        --generated-dir outputs/final_test_lora500_2gpu \
        --output results/validation_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .metrics import (
    ImageMetrics,
    CLIPSimilarity,
)


def evaluate(
    metadata_path: str,
    generated_dir: str,
    output_csv: str,
) -> None:
    """
    Run evaluation on generated images.

    Pipeline:
        Load test metadata
            ↓
        Load generated outputs
            ↓
        Load target images
            ↓
        Calculate metrics (LPIPS, SSIM, CLIP)
            ↓
        Create per-image CSV
            ↓
        Print summary statistics

    Args:
        metadata_path: Path to the metadata CSV with pair_id, target_path columns.
        generated_dir: Directory containing generated images named {pair_id}.png.
        output_csv: Path to save the per-image results CSV.
    """

    metadata = pd.read_csv(
        metadata_path
    )

    metrics = ImageMetrics()
    clip = CLIPSimilarity()

    generated_dir_path = Path(
        generated_dir
    )

    rows = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Evaluating",
    ):

        pair_id = row["pair_id"]

        generated_path = (
            generated_dir_path
            / f"{pair_id}.png"
        )

        target_path = Path(
            row["target_path"]
        )

        if not generated_path.exists():
            continue

        if not target_path.exists():
            continue

        generated = metrics.load_image(
            generated_path
        )

        target = metrics.load_image(
            target_path
        )

        rows.append({
            "pair_id": pair_id,
            "lpips": metrics.lpips(
                generated,
                target,
            ),
            "ssim": metrics.ssim(
                generated,
                target,
            ),
            "clip": clip.similarity(
                generated,
                target,
            ),
        })

    results = pd.DataFrame(rows)

    output_path = Path(
        output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n"
        "Per-image results (first 5):"
    )
    print(results.head())

    print(
        "\n"
        "Mean scores:"
    )
    print(
        results[
            ["lpips", "ssim", "clip"]
        ].mean()
    )

    print(
        f"\nTotal evaluated: {len(results)}"
    )

    print(
        f"\nSaved: {output_path}"
    )


def main() -> None:
    """CLI entry point for evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate comic style transfer quality."
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata CSV with pair_id and target_path.",
    )

    parser.add_argument(
        "--generated-dir",
        required=True,
        help="Directory containing generated {pair_id}.png images.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save per-image results CSV.",
    )

    args = parser.parse_args()

    evaluate(
        metadata_path=args.metadata,
        generated_dir=args.generated_dir,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
