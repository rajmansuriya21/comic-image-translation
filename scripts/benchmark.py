"""
Benchmark script for inference latency and VRAM usage.

Usage:
    python scripts/benchmark.py

Measures:
    - Inference latency (seconds per image)
    - Peak allocated VRAM (GB)
    - Peak reserved VRAM (GB)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

# Add project root to path for imports.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import ComicPipeline


def main() -> None:
    """Run a single-image benchmark and print results."""

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

    input_path = (
        ROOT / "examples" / "input_01.png"
    )

    if not input_path.exists():
        print(
            f"Error: {input_path} not found. "
            "Place an example image in examples/."
        )
        return

    image = Image.open(
        input_path
    ).convert("RGB")

    # Warm-up run (not measured).
    print("Warm-up run...")
    _ = model.generate(
        image=image,
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        strength=0.45,
        guidance_scale=7.0,
        steps=20,
        seed=42,
        resolution=768,
    )

    # Clear memory stats before benchmark.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("Benchmark run...")

    start = time.perf_counter()

    _ = model.generate(
        image=image,
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        strength=0.45,
        guidance_scale=7.0,
        steps=20,
        seed=42,
        resolution=768,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = (
        time.perf_counter() - start
    )

    print(
        "\n"
        "========================================\n"
        "        BENCHMARK RESULTS\n"
        "========================================"
    )

    print(
        f"\nLatency: {latency:.2f} seconds/image"
    )

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        print(
            f"Peak allocated VRAM: "
            f"{allocated:.2f} GB"
        )

        print(
            f"Peak reserved VRAM: "
            f"{reserved:.2f} GB"
        )

        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    else:
        print("Running on CPU (no VRAM stats).")

    print(
        "\n"
        f"Resolution: 768x768\n"
        f"Precision: FP16\n"
        f"Steps: 20\n"
        f"Strength: 0.45\n"
        f"Guidance: 7.0"
    )


if __name__ == "__main__":
    main()
