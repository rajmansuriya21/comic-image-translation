"""
Clone the pinned Diffusers repository for training.

Usage:
    python scripts/setup_training_repo.py

This clones Diffusers v0.37.1 into external/diffusers/ so that
the SDXL LoRA training script is available for reproducibility.
"""

from pathlib import Path
import subprocess


ROOT = (
    Path(__file__).resolve().parents[1]
)

REPO_DIR = (
    ROOT / "external" / "diffusers"
)

REPO_URL = (
    "https://github.com/huggingface/diffusers.git"
)

TAG = "v0.37.1"


def main() -> None:
    """Clone the pinned Diffusers repository."""

    if REPO_DIR.exists():
        print(
            "Diffusers repository already exists:"
        )
        print(REPO_DIR)
        return

    REPO_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            TAG,
            "--depth",
            "1",
            REPO_URL,
            str(REPO_DIR),
        ],
        check=True,
    )

    print(
        "Training repository cloned:"
    )

    print(REPO_DIR)


if __name__ == "__main__":
    main()
