"""
Evaluation metrics for image-to-image style transfer.

Provides reusable metric classes:
    - ImageMetrics: LPIPS and SSIM computation
    - CLIPSimilarity: CLIP-based image-to-image semantic similarity

Usage:
    from evaluation.metrics import ImageMetrics, CLIPSimilarity

    metrics = ImageMetrics()
    lpips_score = metrics.lpips(generated, target)
    ssim_score = metrics.ssim(generated, target)

    clip = CLIPSimilarity()
    clip_score = clip.similarity(generated, target)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from skimage.metrics import structural_similarity
import lpips


class ImageMetrics:
    """
    Computes LPIPS and SSIM between generated and target images.

    LPIPS (Learned Perceptual Image Patch Similarity):
        Lower = better perceptual similarity.
        Uses AlexNet backbone.

    SSIM (Structural Similarity Index):
        Higher = better structural similarity.
        Operates on normalized [0, 1] pixel values.
    """

    def __init__(self) -> None:

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.lpips_model = lpips.LPIPS(
            net="alex"
        ).to(self.device)

        self.lpips_model.eval()

    @staticmethod
    def load_image(
        path,
        size=(768, 768),
    ) -> Image.Image:
        """Load and resize an image to the target size."""

        return (
            Image.open(path)
            .convert("RGB")
            .resize(size)
        )

    @staticmethod
    def _lpips_tensor(
        image: Image.Image,
    ) -> torch.Tensor:
        """Convert a PIL image to a [-1, 1] normalized tensor for LPIPS."""

        array = (
            np.asarray(image)
            .astype(np.float32)
            / 255.0
        )

        tensor = torch.from_numpy(
            array.transpose(2, 0, 1)
        ).unsqueeze(0)

        return tensor * 2.0 - 1.0

    def lpips(
        self,
        generated: Image.Image,
        target: Image.Image,
    ) -> float:
        """
        Compute LPIPS between generated and target images.

        Args:
            generated: Generated comic image.
            target: Ground-truth target comic image.

        Returns:
            LPIPS score (lower is better).
        """

        generated_tensor = (
            self._lpips_tensor(generated)
            .to(self.device)
        )

        target_tensor = (
            self._lpips_tensor(target)
            .to(self.device)
        )

        with torch.no_grad():

            score = self.lpips_model(
                generated_tensor,
                target_tensor,
            )

        return float(
            score.item()
        )

    @staticmethod
    def ssim(
        generated: Image.Image,
        target: Image.Image,
    ) -> float:
        """
        Compute SSIM between generated and target images.

        Args:
            generated: Generated comic image.
            target: Ground-truth target comic image.

        Returns:
            SSIM score (higher is better).
        """

        generated_array = (
            np.asarray(generated)
            .astype(np.float32)
            / 255.0
        )

        target_array = (
            np.asarray(target)
            .astype(np.float32)
            / 255.0
        )

        return float(
            structural_similarity(
                generated_array,
                target_array,
                channel_axis=2,
                data_range=1.0,
            )
        )


class CLIPSimilarity:
    """
    Computes CLIP-based image-to-image cosine similarity.

    Higher = better semantic/image-domain alignment.
    Uses OpenAI CLIP ViT-B/32.
    """

    def __init__(self) -> None:

        from transformers import (
            CLIPModel,
            CLIPProcessor,
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        )

        self.model = (
            CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            .to(self.device)
        )

        self.model.eval()

    def similarity(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
    ) -> float:
        """
        Compute cosine similarity between CLIP embeddings of two images.

        Args:
            image_a: First image (typically generated).
            image_b: Second image (typically target).

        Returns:
            Cosine similarity score (higher is better).
        """

        inputs_a = self.processor(
            images=image_a,
            return_tensors="pt",
        )

        inputs_b = self.processor(
            images=image_b,
            return_tensors="pt",
        )

        pixel_a = (
            inputs_a["pixel_values"]
            .to(self.device)
        )

        pixel_b = (
            inputs_b["pixel_values"]
            .to(self.device)
        )

        with torch.no_grad():

            output_a = self.model.vision_model(
                pixel_values=pixel_a
            )

            output_b = self.model.vision_model(
                pixel_values=pixel_b
            )

            embedding_a = (
                self.model.visual_projection(
                    output_a.pooler_output
                )
            )

            embedding_b = (
                self.model.visual_projection(
                    output_b.pooler_output
                )
            )

        embedding_a = F.normalize(
            embedding_a,
            p=2,
            dim=-1,
        )

        embedding_b = F.normalize(
            embedding_b,
            p=2,
            dim=-1,
        )

        return float(
            torch.sum(
                embedding_a * embedding_b,
                dim=-1,
            ).item()
        )
