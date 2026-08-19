"""
SDXL Img2Img pipeline with the trained comic LoRA adapter.

Usage:
    from inference import ComicPipeline

    pipeline = ComicPipeline(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        lora_path="models/comic_sdxl_lora_500.safetensors",
    )

    result = pipeline.generate(
        image=input_image,
        prompt="comic book illustration ...",
        negative_prompt="photorealistic ...",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline


class ComicPipeline:
    """
    SDXL Img2Img pipeline with the trained comic LoRA adapter.

    This class handles:
        1. Loading the base SDXL model
        2. Loading the fine-tuned LoRA weights
        3. Image preprocessing (RGB conversion, resizing)
        4. Running Img2Img inference
        5. Returning the generated comic-style image
    """

    def __init__(
        self,
        base_model: str,
        lora_path: str,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the comic generation pipeline.

        Args:
            base_model: HuggingFace model ID or path for SDXL base.
            lora_path: Path to the trained LoRA .safetensors file.
            device: Target device ('cuda' or 'cpu'). Auto-detected if None.
        """

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA GPU is available."
            )

        lora_path_obj = Path(lora_path)

        if not lora_path_obj.exists():
            raise FileNotFoundError(
                f"LoRA checkpoint not found: {lora_path_obj}"
            )

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }

        if self.device == "cuda":
            load_kwargs["variant"] = "fp16"

        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            base_model,
            **load_kwargs,
        )

        if self.device == "cuda":
            # Memory-efficient loading: offload unused layers to CPU.
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
        else:
            self.pipe.to(self.device)

        self.pipe.load_lora_weights(
            str(lora_path_obj)
        )

    @staticmethod
    def preprocess(
        image: Image.Image,
        resolution: int = 768,
    ) -> Image.Image:
        """
        Preprocess an input image for inference.

        Args:
            image: Input PIL image.
            resolution: Target resolution (square).

        Returns:
            Preprocessed PIL image in RGB at target resolution.
        """

        if image is None:
            raise ValueError(
                "Input image is required."
            )

        image = image.convert("RGB")

        return image.resize(
            (resolution, resolution)
        )

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        strength: float = 0.45,
        guidance_scale: float = 7.0,
        steps: int = 20,
        seed: int = 42,
        resolution: int = 768,
    ) -> Image.Image:
        """
        Generate a comic-style image from a realistic input.

        Pipeline:
            Input Image
                ↓
            RGB Conversion
                ↓
            768×768 Resize
                ↓
            Latent Encoding
                ↓
            Diffusion Denoising (with LoRA adaptation)
                ↓
            VAE Decode
                ↓
            Comic Output

        Args:
            image: Input realistic face image.
            prompt: Text prompt describing the desired comic style.
            negative_prompt: Text prompt describing what to avoid.
            strength: How much to transform the input (0.0–1.0).
            guidance_scale: Classifier-free guidance scale.
            steps: Number of denoising steps.
            seed: Random seed for reproducibility.
            resolution: Target resolution in pixels.

        Returns:
            Generated comic-style PIL image.
        """

        image = self.preprocess(
            image,
            resolution=resolution,
        )

        generator = torch.Generator(
            device=self.device
        ).manual_seed(int(seed))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            generator=generator,
        )

        return result.images[0].convert("RGB")
