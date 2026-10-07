#!/usr/bin/env python3
"""
SOTA Deepfake Training Engine
Trains the Dual-Stream Spatial-Frequency Cross-Attention Architecture
using Self-Blended Image (SBI) dynamic augmentation for maximum generalizability.
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Ensure UTF-8 stdout encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.sota_model import SOTADualStreamDetector
from training.dataset import SOTADeepfakeDataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training Batch", leave=False)
    for x_rgb, x_freq, labels in pbar:
        x_rgb = x_rgb.to(device)
        x_freq = x_freq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(x_rgb, x_freq)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * x_rgb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/max(1,total):.1f}%"})

    return running_loss / max(1, total), 100.0 * correct / max(1, total)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_rgb, x_freq, labels in loader:
            x_rgb = x_rgb.to(device)
            x_freq = x_freq.to(device)
            labels = labels.to(device)

            logits = model(x_rgb, x_freq)
            loss = criterion(logits, labels)

            running_loss += loss.item() * x_rgb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = 100.0 * correct / max(1, total)
    loss = running_loss / max(1, total)
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train SOTA Dual-Stream Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Dataset directory")
    parser.add_argument("--val-dir", type=str, default="data/validation", help="Validation dataset directory")
    parser.add_argument("--sbi-samples", type=int, default=1000, help="Number of dynamic SBI synthetic samples per epoch")
    parser.add_argument("--output-model", type=str, default="models/sota_model.pth", help="Target checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print("  VERITAS // SOTA DUAL-STREAM DEEPFAKE TRAINING ENGINE")
    print("=" * 65)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Architecture   : SOTADualStreamDetector (Spatial SRM + Freq + Cross-Attn)")
    print(f"[*] Target Model   : {args.output_model}")
    print(f"[*] Hyperparams    : Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")
    print("-" * 65)

    # 1. Dataset & DataLoader
    print("[*] Initializing SOTA Multi-Modal Dataset (Real Data + Dynamic SBI Synthesis)...")
    train_dataset = SOTADeepfakeDataset(
        data_dir=args.data_dir,
        use_sbi_synthesis=True,
        num_synthetic_samples=args.sbi_samples
    )

    if len(train_dataset) == 0:
        print("[!] No training samples found. Please ensure data/train exists or assets contains images.", file=sys.stderr)
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows safe
        pin_memory=True if device.type == 'cuda' else False
    )

    # Validation DataLoader
    val_loader = None
    if Path(args.val_dir).exists():
        val_dataset = SOTADeepfakeDataset(data_dir=args.val_dir, use_sbi_synthesis=False)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print(f"[+] Loaded {len(val_dataset)} validation evaluation images.")

    print(f"[+] Loaded {len(train_dataset)} training pairs.")

    # 2. Model, Loss, Optimizer
    print("[*] Building SOTA Dual-Stream Network...")
    model = SOTADualStreamDetector(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 3. Training Loop
    best_acc = 0.0
    Path("models").mkdir(exist_ok=True)

    print("\n[*] Starting training loop...")
    for epoch in range(args.epochs):
        t_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        t_epoch = time.time() - t_start

        current_lr = scheduler.get_last_lr()[0]
        
        # Validation if available
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"  Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} | Time: {t_epoch:.1f}s")
            score_to_compare = val_acc
        else:
            print(f"  Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {current_lr:.6f} | Time: {t_epoch:.1f}s")
            score_to_compare = train_acc

        if score_to_compare > best_acc:
            best_acc = score_to_compare
            torch.save(model.state_dict(), args.output_model)
            # Also update best_model.pth so server and CLI immediately benefit
            torch.save(model.spatial_backbone.state_dict(), "models/best_model.pth")
            print(f"    [+] New best model saved to: {args.output_model} (Score: {best_acc:.2f}%)")

    print("\n" + "=" * 65)
    print(f"[+] Training Complete! Best Score: {best_acc:.2f}%")
    print(f"[+] Checkpoint Ready: {args.output_model} & models/best_model.pth")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
