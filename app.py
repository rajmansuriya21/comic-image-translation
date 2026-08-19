"""
Gradio application for realistic image to comic style conversion.

Usage:
    python app.py

The application loads the SDXL base model with the trained LoRA-500
adapter and provides an interactive UI for image-to-comic conversion.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import yaml

from inference import ComicPipeline


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)


MODEL_PATH = ROOT_DIR / config["lora_path"]


# ============================================================
# LOAD MODEL ONCE
# ============================================================

generator = ComicPipeline(
    base_model=config["base_model"],
    lora_path=str(MODEL_PATH),
)


# ============================================================
# GENERATION
# ============================================================

def generate_comic(
    image,
    strength,
    guidance_scale,
    steps,
    seed,
):
    """
    Generate a comic-style image from a realistic face input.

    Called by the Gradio UI when the user clicks 'Generate Comic'.
    """

    if image is None:
        raise gr.Error(
            "Please upload an input image."
        )

    try:
        result = generator.generate(
            image=image,
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"],
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            seed=int(seed),
            resolution=int(config["resolution"]),
        )

        return result

    except torch.cuda.OutOfMemoryError:
        raise gr.Error(
            "GPU memory exhausted. "
            "Try reducing the number of inference steps."
        )

    except Exception as exc:
        raise gr.Error(
            f"Generation failed: {exc}"
        )


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Realistic Image → Comic",
) as demo:

    gr.Markdown(
        """
        # 🎨 Realistic Image → Comic Style

        ### SDXL + LoRA-500

        Upload a realistic image and convert it into
        modern western comic-book artwork.

        **Final test improvement:**
        LPIPS ↓ 12.03% | SSIM ↑ 2.59% | CLIP ↑ 26.43%
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Input Image",
            )

            strength = gr.Slider(
                minimum=0.20,
                maximum=0.80,
                value=config["strength"],
                step=0.05,
                label="Style Strength",
            )

            guidance = gr.Slider(
                minimum=3.0,
                maximum=10.0,
                value=config["guidance_scale"],
                step=0.5,
                label="Guidance Scale",
            )

            steps = gr.Slider(
                minimum=15,
                maximum=30,
                value=config["num_inference_steps"],
                step=1,
                label="Inference Steps",
            )

            seed = gr.Number(
                value=config["seed"],
                precision=0,
                label="Seed",
            )

            generate_button = gr.Button(
                "Generate Comic",
                variant="primary",
            )

        with gr.Column():

            output_image = gr.Image(
                type="pil",
                label="Comic Output",
            )

    generate_button.click(
        fn=generate_comic,
        inputs=[
            input_image,
            strength,
            guidance,
            steps,
            seed,
        ],
        outputs=[
            output_image,
        ],
    )


if __name__ == "__main__":
    demo.launch()
