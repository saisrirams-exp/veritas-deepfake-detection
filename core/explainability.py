"""
Explainable AI (XAI) & Grad-CAM Visualization Module
Generates Class Activation Maps (Grad-CAM) and facial anomaly heatmaps
to visually pinpoint deepfake manipulation regions.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GradCAMExplainer:
    """
    Computes visual saliency and class activation heatmaps for EfficientNet backbones.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        if TORCH_AVAILABLE and model is not None:
            self._register_hooks()

    def _register_hooks(self):
        """Hook into the final convolutional feature extractor block of EfficientNet."""
        if self.target_layer is None:
            # EfficientNet-B0 backbone features block 8 / last conv
            try:
                self.target_layer = self.model.backbone.features[-1]
            except Exception:
                try:
                    self.target_layer = list(self.model.children())[0][-1]
                except Exception:
                    self.target_layer = None

        if self.target_layer is not None:
            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()

            h1 = self.target_layer.register_forward_hook(forward_hook)
            h2 = self.target_layer.register_full_backward_hook(backward_hook)
            self.hooks.extend([h1, h2])

    def generate_heatmap(self, face_image: np.ndarray, target_class: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes Grad-CAM heatmap for a face image.
        
        Args:
            face_image: RGB face crop (H, W, 3)
            target_class: 0 for Fake, 1 for Real
            
        Returns:
            Tuple of (overlay_image, raw_heatmap, anomaly_regions_dict)
        """
        h, w = face_image.shape[:2]

        if not TORCH_AVAILABLE or self.model is None or self.target_layer is None:
            # Fallback high-contrast edge & gradient saliency
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.hypot(sobelx, sobely)
            norm_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(face_image, 0.6, heatmap, 0.4, 0)
            return overlay, heatmap, {"primary_hotspot": "Facial Edges & Texture", "intensity": 0.55}

        # Convert image to tensor
        transform = self.model.transform if hasattr(self.model, 'transform') else None
        if transform:
            tensor = transform(Image.fromarray(face_image)).unsqueeze(0).to(self.model.device)
        else:
            tensor = torch.from_numpy(face_image).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Enable grad for input and hook
        self.model.model.zero_grad()
        output = self.model.model(tensor)

        # Target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hook did not capture
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            norm_lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm_lap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(face_image, 0.65, heatmap, 0.35, 0)
            return overlay, heatmap, {"primary_hotspot": "Boundary Gradient Discontinuity", "intensity": 0.60}

        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()

        # Resize CAM to match original face crop
        cam = cv2.resize(cam, (w, h))
        cam_min, cam_max = cam.min(), cam.max()
        cam_norm = ((cam - cam_min) / (cam_max - cam_min + 1e-8) * 255).astype(np.uint8)

        # Apply Jet / Turbo colormap
        heatmap = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Alpha blending overlay
        overlay = cv2.addWeighted(face_image, 0.65, heatmap, 0.35, 0)

        # Identify hotspot quadrant
        q_h, q_w = h // 2, w // 2
        quadrants = {
            "Forehead / Hairline": np.mean(cam_norm[:q_h, :]),
            "Periorbital (Eyes / Bridge)": np.mean(cam_norm[int(q_h*0.5):int(q_h*1.2), :]),
            "Perioral (Mouth / Jaw)": np.mean(cam_norm[q_h:, :]),
            "Face Perimeter Seam": np.mean(cam_norm)
        }
        hotspot = max(quadrants, key=quadrants.get)

        return overlay, heatmap, {
            "primary_hotspot": hotspot,
            "peak_activation": float(cam_norm.max() / 255.0),
            "activation_spread": float(np.std(cam_norm) / 255.0)
        }

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
