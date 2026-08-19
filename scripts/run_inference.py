"""
CLI inference script for comic style transfer.

Runs inference on a single image without launching Gradio.

Usage:
    python scripts/run_inference.py \
        --input examples/input_01.png \
        --output outputs/example_output.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

# Add project root to path for imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import ComicPipeline


def main() -> None:
    """Run inference on a single image."""

    parser = argparse.ArgumentParser(
        description="Generate a comic-style image from a realistic input."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the generated comic image.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20).",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.45,
        help="Style strength 0.0–1.0 (default: 0.45).",
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="Guidance scale (default: 7.0).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    with open(
        ROOT / "configs" / "config.yaml",
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    print("Loading model...")

    model = ComicPipeline(
        base_model=config["base_model"],
        lora_path=str(
            ROOT / config["lora_path"]
        ),
    )

    image = Image.open(
        args.input
    ).convert("RGB")

    print(f"Generating comic from: {args.input}")

    output = model.generate(
        image=image,
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        strength=args.strength,
        guidance_scale=args.guidance,
        steps=args.steps,
        seed=args.seed,
        resolution=config["resolution"],
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.save(
        output_path
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
