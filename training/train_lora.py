"""
LoRA fine-tuning script for SDXL comic style transfer.

This script automates the full training pipeline:
    1. Clone the pinned Diffusers repository
    2. Prepare the training command
    3. Launch distributed training with Accelerate

Usage:
    python training/train_lora.py

Prerequisites:
    - Dataset prepared in data/lora_train/ (Diffusers ImageFolder format)
    - 2 GPUs available
    - PyTorch with CUDA support

The final LoRA-500 adapter was trained with this exact configuration
on 2 NVIDIA GPUs using NCCL distributed backend.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT_DIR = Path(
    __file__
).resolve().parents[1]


DIFFUSERS_DIR = (
    ROOT_DIR / "external" / "diffusers"
)

DIFFUSERS_REPO = (
    "https://github.com/huggingface/diffusers.git"
)

DIFFUSERS_TAG = "v0.37.1"

TRAINING_SCRIPT = (
    DIFFUSERS_DIR
    / "examples"
    / "text_to_image"
    / "train_text_to_image_lora_sdxl.py"
)


TRAIN_DATA = (
    ROOT_DIR
    / "data"
    / "lora_train"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "models"
    / "training_output"
)


def run_command(command: list[str]) -> None:
    """Execute a shell command and print it for reproducibility."""

    print(
        "\nRunning:\n",
        " ".join(command),
        "\n",
    )

    subprocess.run(
        command,
        check=True,
    )


def prepare_diffusers() -> None:
    """Clone the pinned Diffusers repository for the training script."""

    DIFFUSERS_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DIFFUSERS_DIR.exists():

        run_command([
            "git",
            "clone",
            "--branch",
            DIFFUSERS_TAG,
            "--depth",
            "1",
            DIFFUSERS_REPO,
            str(DIFFUSERS_DIR),
        ])


def main() -> None:
    """Run the full LoRA fine-tuning pipeline."""

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_DATA}\n"
            f"Please prepare the dataset in Diffusers ImageFolder format.\n"
            f"See data/DATASET.md for instructions."
        )

    prepare_diffusers()

    if not TRAINING_SCRIPT.exists():
        raise FileNotFoundError(
            f"Training script not found: {TRAINING_SCRIPT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # fmt: off
    command = [
        "accelerate",
        "launch",

        "--num_processes=2",
        "--num_machines=1",
        "--mixed_precision=fp16",

        str(TRAINING_SCRIPT),

        "--pretrained_model_name_or_path",
        "stabilityai/stable-diffusion-xl-base-1.0",

        "--train_data_dir",
        str(TRAIN_DATA),

        "--caption_column",
        "text",

        "--resolution",
        "768",

        "--center_crop",
        "--random_flip",

        "--train_batch_size",
        "1",

        "--gradient_accumulation_steps",
        "4",

        "--gradient_checkpointing",

        "--learning_rate",
        "1e-4",

        "--lr_scheduler",
        "constant",

        "--lr_warmup_steps",
        "0",

        "--max_train_steps",
        "500",

        "--checkpointing_steps",
        "250",

        "--mixed_precision",
        "fp16",

        "--use_8bit_adam",

        "--seed",
        "42",

        "--output_dir",
        str(OUTPUT_DIR),
    ]
    # fmt: on

    run_command(command)


if __name__ == "__main__":
    main()
