"""
VERITAS Weight Initialization Utility
Initializes local model weight checkpoints (models/sota_model.pth & models/best_model.pth)
for the VERITAS Cyber-Forensics Engine.
"""

import sys
from pathlib import Path

def main():
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    except ImportError:
        print("[!] PyTorch and torchvision are required. Please run: pip install -r requirements.txt")
        sys.exit(1)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_model.pth"

    if best_model_path.exists():
        print(f"[i] Model weights already exist at: {best_model_path} ({best_model_path.stat().st_size / (1024*1024):.2f} MB)")
        return

    print("[*] Initializing VERITAS Spatial Backbone weights...")
    try:
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        print("[*] Loaded pre-trained ImageNet baseline weights.")
    except Exception as e:
        print(f"[!] Warning: Could not fetch online weights ({e}). Initializing randomly.")
        backbone = efficientnet_b0(weights=None)

    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 2)
    )

    torch.save(backbone.state_dict(), best_model_path)
    print(f"[+] Spatial weights saved successfully to: {best_model_path} ({best_model_path.stat().st_size / (1024*1024):.2f} MB)")
    print("[+] Platform ready. Launch the studio with: python server.py or .\\run_website.bat")

if __name__ == "__main__":
    main()
