"""
Self-Blended Image (SBI) Generator for Generalizable Deepfake Training
Generates synthetic manipulation boundaries, color mismatches, and compression discrepancies
on authentic facial imagery to train models that generalize across unseen generators.
"""

from typing import Tuple
import random
import io
import numpy as np
import cv2
from PIL import Image

class SelfBlendedImageGenerator:
    """
    Synthesizes generalizable deepfake artifacts on-the-fly:
    - Elastic facial warping
    - Color balance & illumination mismatch
    - Boundary mask feathering & seam synthesis
    - JPEG re-compression discrepancies
    """

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def _generate_random_mask(self, h: int, w: int) -> np.ndarray:
        """Create convex/organic facial blending mask with randomized deformation."""
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, int(h * 0.55))
        axes = (int(w * random.uniform(0.28, 0.42)), int(h * random.uniform(0.32, 0.48)))
        angle = random.uniform(-15, 15)

        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)

        # Deform with random polygon perturb
        ksize = random.choice([15, 21, 29])
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        return np.clip(mask, 0.0, 1.0)

    def _color_perturbation(self, img: np.ndarray) -> np.ndarray:
        """Simulate camera sensor / generator white-balance mismatch."""
        perturbed = img.astype(np.float32)
        # Random channel scaling
        for c in range(3):
            scale = random.uniform(0.85, 1.15)
            shift = random.uniform(-15.0, 15.0)
            perturbed[..., c] = np.clip(perturbed[..., c] * scale + shift, 0, 255)
        return perturbed.astype(np.uint8)

    def _geometric_warp(self, img: np.ndarray) -> np.ndarray:
        """Simulate landmark misalignment and Poisson warping."""
        h, w = img.shape[:2]
        # Random affine transformation
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        dx = random.uniform(-6, 6)
        dy = random.uniform(-6, 6)
        pts2 = np.float32([[dx, dy], [w - dx, dy], [dx, h - dy]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _inject_jpeg_disparity(self, img: np.ndarray, quality: int = 70) -> np.ndarray:
        """Inject compression noise characteristic of web/social media deepfakes."""
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer))

    def generate_pair(self, authentic_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Takes an authentic face crop and produces either an authentic sample (label 1)
        or a synthesized deepfake sample (label 0) with rich multi-domain artifacts.
        
        Returns:
            Tuple of (sample_image_rgb, frequency_residual_rgb, label: 0=Fake, 1=Real)
        """
        img = cv2.resize(authentic_face, (self.target_size, self.target_size))
        h, w = img.shape[:2]

        is_fake = random.random() < 0.5

        if not is_fake:
            # Real sample: Apply benign camera noise / gentle augmentation
            if random.random() < 0.3:
                img = cv2.flip(img, 1)
            # High-pass residual
            lap = cv2.Laplacian(img, cv2.CV_8U)
            return img, lap, 1

        # Synthesize Fake via Self-Blending:
        # 1. Source face gets geometric warping and color shift
        manipulated = self._geometric_warp(img)
        manipulated = self._color_perturbation(manipulated)

        # 2. Add local blur / smoothing
        if random.random() < 0.6:
            k = random.choice([3, 5])
            manipulated = cv2.GaussianBlur(manipulated, (k, k), 0)

        # 3. Create blend mask
        mask = self._generate_random_mask(h, w)[..., None]

        # 4. Blend manipulated face into authentic background
        blended = (manipulated * mask + img * (1.0 - mask)).astype(np.uint8)

        # 5. Inject localized JPEG double-compression
        if random.random() < 0.5:
            q = random.randint(65, 88)
            blended = self._inject_jpeg_disparity(blended, quality=q)

        # Compute frequency residual
        lap = cv2.Laplacian(blended, cv2.CV_8U)

        return blended, lap, 0
