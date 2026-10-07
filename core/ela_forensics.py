"""
Error Level Analysis (ELA) & Compression Residual Forensics Module
Identifies double compression artifacts, splicing edges, and sensor noise inconsistencies.
"""

from typing import Dict, Any, Tuple
import io
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance

class ELAForensicAnalyzer:
    """
    Examines compression disparities between the face region and surrounding background.
    Detects re-saving anomalies, splicing artifacts, and local quality inconsistencies.
    """

    def __init__(self, quality: int = 90, scale: float = 12.0):
        self.quality = quality
        self.scale = scale

    def compute_ela(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute standard Error Level Analysis map.
        
        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
            
        Returns:
            Tuple of (diff_image_rgb, colorized_heatmap, mean_error)
        """
        pil_img = Image.fromarray(image_rgb)
        
        # Save to memory buffer at specific JPEG quality
        buffer = io.BytesIO()
        pil_img.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        # Compute absolute difference
        ela_diff = ImageChops.difference(pil_img, recompressed)
        
        # Scale difference for visibility
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0:
            max_diff = 1
        scale_factor = 255.0 / max_diff

        enhancer = ImageEnhance.Brightness(ela_diff)
        ela_enhanced = enhancer.enhance(min(scale_factor, self.scale))
        ela_np = np.array(ela_enhanced)

        # Convert to single channel energy map
        gray_diff = cv2.cvtColor(np.array(ela_diff), cv2.COLOR_RGB2GRAY)
        mean_err = float(np.mean(gray_diff))

        # Colorize ELA for forensic display (Turbo / Jet)
        norm_diff = cv2.normalize(gray_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        return ela_np, heatmap, mean_err

    def analyze(self, image_rgb: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Computes comparative ELA metrics between the Face Region and the Outer Image Context.
        
        Args:
            image_rgb: Full RGB image
            face_box: (x, y, w, h) bounding box of face
            
        Returns:
            Forensic metrics highlighting compression disparity and blending seams.
        """
        h, w = image_rgb.shape[:2]
        ela_vis, ela_heatmap, global_mean = self.compute_ela(image_rgb)

        fx, fy, fw, fh = face_box
        fx2, fy2 = min(w, fx + fw), min(h, fy + fh)

        # Extract face ELA vs background ELA
        ela_gray = cv2.cvtColor(ela_vis, cv2.COLOR_RGB2GRAY)
        face_ela = ela_gray[fy:fy2, fx:fx2]
        
        # Create mask of background (outside face box)
        bg_mask = np.ones((h, w), dtype=bool)
        bg_mask[fy:fy2, fx:fx2] = False
        bg_ela = ela_gray[bg_mask]

        face_mean = float(np.mean(face_ela)) if face_ela.size > 0 else global_mean
        bg_mean = float(np.mean(bg_ela)) if bg_ela.size > 0 else global_mean
        face_std = float(np.std(face_ela)) if face_ela.size > 0 else 0.0

        # Disparity ratio: In deepfakes, face has different compression generation than body/background
        disparity_ratio = float(abs(face_mean - bg_mean) / (bg_mean + 1e-4))

        # Boundary seam artifact analysis: measure edge gradient along the face perimeter
        margin = max(4, int(min(fw, fh) * 0.08))
        inner_y1, inner_y2 = min(h, fy + margin), max(0, fy2 - margin)
        inner_x1, inner_x2 = min(w, fx + margin), max(0, fx2 - margin)

        boundary_mask = np.zeros((h, w), dtype=np.uint8)
        boundary_mask[fy:fy2, fx:fx2] = 255
        if inner_y2 > inner_y1 and inner_x2 > inner_x1:
            boundary_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

        boundary_ela = ela_gray[boundary_mask > 0]
        boundary_activity = float(np.mean(boundary_ela)) if boundary_ela.size > 0 else face_mean

        # Anomaly score calculation
        ela_anomaly_score = 0.0
        if disparity_ratio > 0.35:
            ela_anomaly_score += 0.45
        elif disparity_ratio > 0.20:
            ela_anomaly_score += 0.25

        if boundary_activity > face_mean * 1.30:
            # High compression error concentrated on the blending boundary
            ela_anomaly_score += 0.35

        ela_anomaly_score = float(np.clip(ela_anomaly_score, 0.05, 0.95))

        return {
            "ela_score": round(ela_anomaly_score, 4),
            "disparity_ratio": round(disparity_ratio, 4),
            "face_ela_mean": round(face_mean, 2),
            "background_ela_mean": round(bg_mean, 2),
            "boundary_activity": round(boundary_activity, 2),
            "face_ela_std": round(face_std, 2),
            "ela_heatmap": ela_heatmap,
            "ela_visual": ela_vis
        }
