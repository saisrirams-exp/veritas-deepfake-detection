"""
Dataset Loaders for SOTA Multi-Modal Deepfake Training
Supports standard directory structures (real/fake), FaceForensics++, Celeb-DF,
and dynamic Self-Blended Image (SBI) data generation with genuine 2D-FFT frequency tensors.
"""

from typing import List, Tuple, Optional
from pathlib import Path
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from core.sota_model import compute_fft_tensor
from training.sbi_generator import SelfBlendedImageGenerator


class SOTADeepfakeDataset(Dataset):
    """
    Multi-modal PyTorch dataset delivering paired RGB spatial crops
    and true 2D Fast Fourier Transform (FFT) log-magnitude spectral tensors.
    """
    def __init__(
        self,
        data_dir: Optional[str] = None,
        use_sbi_synthesis: bool = True,
        num_synthetic_samples: int = 1000,
        img_size: int = 224,
        max_samples: Optional[int] = None
    ):
        self.img_size = img_size
        self.use_sbi = use_sbi_synthesis
        self.sbi_gen = SelfBlendedImageGenerator(target_size=img_size)
        self.samples: List[Tuple[str, int]] = []

        # Standard Spatial RGB Normalization
        self.transform_rgb = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load filepaths if directory provided
        if data_dir and Path(data_dir).exists():
            d = Path(data_dir)
            # 1. Standard fake / real subdirectories
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                for p in (d / 'fake').glob(ext):
                    self.samples.append((str(p), 0))
                for p in (d / 'real').glob(ext):
                    self.samples.append((str(p), 1))

            # 2. Check for Celeb-DF / FF++ naming patterns if nested
            for sub in d.glob("*"):
                if sub.is_dir() and sub.name not in ('fake', 'real'):
                    label = 0 if any(k in sub.name.lower() for k in ('fake', 'manipulated', 'synthesis', 'df', 'f2f', 'fs', 'nt')) else 1
                    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
                        for p in sub.glob(ext):
                            self.samples.append((str(p), label))

        # Shuffle sample list
        random.seed(42)
        random.shuffle(self.samples)

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        # Also discover real sample images in assets/
        assets_dir = Path("assets")
        self.authentic_seeds = []
        if assets_dir.exists():
            for p in assets_dir.glob("*.jpg"):
                if "authentic" in p.name.lower() or "real" in p.name.lower():
                    self.authentic_seeds.append(str(p))

        self.num_synthetic = num_synthetic_samples if (self.use_sbi and self.authentic_seeds) else 0

    def __len__(self) -> int:
        return len(self.samples) + self.num_synthetic

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Dynamic SBI sample generation
        if idx >= len(self.samples):
            seed_path = random.choice(self.authentic_seeds)
            seed_img = cv2.cvtColor(cv2.imread(seed_path), cv2.COLOR_BGR2RGB)
            img_rgb, _, label = self.sbi_gen.generate_pair(seed_img)
            
            t_rgb = self.transform_rgb(Image.fromarray(img_rgb))
            # Compute real 2D FFT tensor
            t_freq = compute_fft_tensor(t_rgb.unsqueeze(0)).squeeze(0)
            return t_rgb, t_freq, torch.tensor(label, dtype=torch.long)

        # Static disk sample
        path, label = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            # Fallback zero tensor
            z = torch.zeros(3, self.img_size, self.img_size)
            return z, z, torch.tensor(label, dtype=torch.long)
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t_rgb = self.transform_rgb(Image.fromarray(rgb))
        # Compute true 2D Fast Fourier Transform tensor
        t_freq = compute_fft_tensor(t_rgb.unsqueeze(0)).squeeze(0)
        return t_rgb, t_freq, torch.tensor(label, dtype=torch.long)
