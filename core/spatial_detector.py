"""
Deep Learning Spatial Forensics Module
Encapsulates deep convolutional and vision transformer architectures for facial artifact recognition.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LightweightDeepfakeDetector(nn.Module if TORCH_AVAILABLE else object):
    """
    EfficientNet-B0 deepfake detection architecture with custom dual-class head.
    Matches the author's architecture for 100% backward compatibility.
    """
    def __init__(self, pretrained: bool = True):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 2)
        )

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """Extract high-level feature maps before classification pooling."""
        return self.backbone.features(x)


class SpatialForensicDetector:
    """
    Manages model loading, device placement, multi-scale face inference,
    and fallback handling.
    """

    def __init__(self, model_path: str = "models/best_model.pth", device: Optional[str] = None):
        self.model_path = Path(model_path)
        
        if TORCH_AVAILABLE:
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.model = self._load_or_init_model()
        else:
            self.device = "cpu"
            self.model = None

    def _load_or_init_model(self):
        """Load trained weights if available; otherwise initialize pre-trained ImageNet backbone."""
        model = LightweightDeepfakeDetector(pretrained=True)
        if self.model_path.exists():
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.is_custom_trained = True
            except Exception as e:
                print(f"[!] Warning loading {self.model_path}: {e}. Running with initialized backbone.")
                self.is_custom_trained = False
        else:
            self.is_custom_trained = False

        model.to(self.device)
        model.eval()
        return model

    def predict(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on face crop.
        
        Args:
            face_image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Dictionary with predicted class, confidence, fake probability, and real probability.
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback if PyTorch is temporarily unavailable
            return {
                "pred_class": 0,
                "fake_prob": 0.50,
                "real_prob": 0.50,
                "confidence": 50.0,
                "label": "UNCERTAIN"
            }

        # Convert to PIL Image
        if isinstance(face_image, np.ndarray):
            pil_image = Image.fromarray(face_image)
        else:
            pil_image = face_image

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        fake_prob = float(probs[0])
        real_prob = float(probs[1])

        # If custom model weights aren't loaded yet, blend with spatial texture variance
        if not self.is_custom_trained:
            # Calibrate using spatial blurriness and edge distortion
            gray = np.array(pil_image.convert('L'))
            lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
            # Heavy smoothing or anomalous sharpness adjusts baseline
            if lap_var < 80.0:  # Excessively smooth/blurry face
                fake_prob = min(0.85, fake_prob + 0.25)
                real_prob = 1.0 - fake_prob

        pred_class = 0 if fake_prob >= 0.5 else 1
        confidence = (fake_prob if pred_class == 0 else real_prob) * 100.0

        return {
            "pred_class": pred_class,
            "label": "FAKE" if pred_class == 0 else "REAL",
            "fake_prob": round(fake_prob, 4),
            "real_prob": round(real_prob, 4),
            "confidence": round(confidence, 2),
            "is_custom_weights": getattr(self, 'is_custom_trained', False)
        }
