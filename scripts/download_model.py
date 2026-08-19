"""
Download the trained LoRA model from Hugging Face Hub.

Usage:
    python scripts/download_model.py

    # With a specific repo:
    python scripts/download_model.py --repo your-username/comic-sdxl-lora

The script downloads the LoRA .safetensors file to models/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Download the LoRA model from Hugging Face Hub."""

    parser = argparse.ArgumentParser(
        description="Download the trained LoRA model."
    )

    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("HF_LORA_REPO", ""),
        help="Hugging Face model repository ID.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="comic_sdxl_lora_500.safetensors",
        help="Filename to download.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "models"),
        help="Directory to save the model.",
    )

    args = parser.parse_args()

    if not args.repo:
        print(
            "Error: No repository specified.\n"
            "\n"
            "Usage:\n"
            "  python scripts/download_model.py "
            "--repo your-username/comic-sdxl-lora\n"
            "\n"
            "Or set the HF_LORA_REPO environment variable."
        )
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Error: huggingface-hub not installed.\n"
            "Run: pip install huggingface-hub"
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    print(f"Downloading {args.filename} from {args.repo}...")

    path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        local_dir=str(output_dir),
        token=token,
    )

    print(f"Downloaded: {path}")


if __name__ == "__main__":
    main()
