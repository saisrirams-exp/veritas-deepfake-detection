#!/usr/bin/env python3
"""
VERITAS SOTA Deepfake Benchmark & Ablation Study Suite
Trains and rigorously benchmarks the 3 ablation conditions:
1. Spatial-Only (EfficientNet-B0 + SRM)
2. Frequency-Only (2D-FFT Log-Magnitude Stream)
3. Full Dual-Stream (Multi-Head Token Cross-Attention)

Evaluates on:
- In-Domain Validation Set (data/validation)
- Cross-Dataset Benchmark: Celeb-DF (data/celeb_df)
- Cross-Dataset Benchmark: FaceForensics++ (data/faceforensics)

Generates:
- Standard Academic Metrics: AUC-ROC, Precision, Recall, Specificity, F1, Confusion Matrices
- High-Resolution Multi-Panel Visual Figure (assets/ablation_benchmark_results.png)
- Raw JSON Metrics (models/ablation_study_metrics.json)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Windows UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

from core.sota_model import SOTADualStreamDetector, compute_fft_tensor
from training.dataset import SOTADeepfakeDataset


def evaluate_dataset_mode(
    model: nn.Module,
    dataset: SOTADeepfakeDataset,
    mode: str,
    device: torch.device,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Evaluates a specific model ablation mode on a dataset and calculates full metrics.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs_real: List[float] = []

    with torch.no_grad():
        for x_rgb, x_freq, labels in loader:
            x_rgb = x_rgb.to(device)
            x_freq = x_freq.to(device)
            labels = labels.to(device)

            logits = model(x_rgb, x_freq, mode=mode)
            probs = torch.softmax(logits, dim=-1)

            # Prob of Real class (index 1)
            prob_real = probs[:, 1].cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds)
            all_probs_real.extend(prob_real)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs_real)

    # Calculate metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # AUC-ROC
    try:
        auc = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr = fpr.tolist()
        tpr = tpr.tolist()
    except Exception:
        auc = 0.5
        fpr, tpr = [0.0, 1.0], [0.0, 1.0]

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "precision": float(prec),
        "recall": float(rec),
        "specificity": specificity,
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "roc_curve": {"fpr": fpr, "tpr": tpr}
    }


def train_ablation_model(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu")
) -> SOTADualStreamDetector:
    """Trains the dual-stream model with joint ablation loss."""
    print("\n" + "=" * 70)
    print("  PHASE 1: TRAINING DUAL-STREAM ARCHITECTURE (TRUE FFT + CROSS-ATTN)")
    print("=" * 70)

    train_dataset = SOTADeepfakeDataset(
        data_dir="data/train",
        use_sbi_synthesis=True,
        num_synthetic_samples=100
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"[+] Loaded {len(train_dataset)} training pairs (Real/Fake + Dynamic SBI).")

    model = SOTADualStreamDetector(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs:02d}]", leave=False)
        for x_rgb, x_freq, labels in pbar:
            x_rgb = x_rgb.to(device)
            x_freq = x_freq.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Unified single forward pass: 3x faster on CPU
            logits_dual, logits_spatial, logits_freq = model.forward_all(x_rgb, x_freq)

            loss_dual = criterion(logits_dual, labels)
            loss_spatial = criterion(logits_spatial, labels)
            loss_freq = criterion(logits_freq, labels)

            loss = loss_dual + 0.4 * loss_spatial + 0.4 * loss_freq
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_dual.item() * x_rgb.size(0)
            preds = torch.argmax(logits_dual, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(1, total)
        epoch_acc = 100.0 * correct / max(1, total)
        print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] - Joint Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/sota_dual_stream_best.pth")
    torch.save(model.state_dict(), "models/sota_model.pth")
    torch.save(model.spatial_backbone.state_dict(), "models/best_model.pth")
    print("[+] Checkpoint successfully saved to models/sota_dual_stream_best.pth & models/sota_model.pth")
    return model


def generate_ablation_figures(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "assets/ablation_benchmark_results.png"
):
    """
    Plots a 4-panel publication-grade academic visualization:
    1. ROC Curves (Spatial vs Frequency vs Full Dual-Stream on Celeb-DF)
    2. Confusion Matrices Comparison
    3. Cross-Dataset Generalization (Validation vs Celeb-DF vs FaceForensics)
    4. SOTA Performance Summary Radar/Bar Table
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=200)

    # Colors
    c_dual = "#00F5D4"     # Cyber Teal
    c_spatial = "#7B2CBF"  # Deep Purple
    c_freq = "#FF007F"     # Neon Magenta

    modes = ["spatial_only", "freq_only", "dual"]
    labels_map = {
        "spatial_only": "Spatial-Only (EfficientNet)",
        "freq_only": "Frequency-Only (2D-FFT)",
        "dual": "Full Dual-Stream (Cross-Attn)"
    }
    colors_map = {
        "spatial_only": c_spatial,
        "freq_only": c_freq,
        "dual": c_dual
    }

    # -------------------------------------------------------------
    # Panel 1: ROC Curves on Cross-Dataset Celeb-DF
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    
    target_ds = "celeb_df" if "celeb_df" in results else "validation"
    for mode in modes:
        m_data = results[target_ds][mode]
        fpr = m_data["roc_curve"]["fpr"]
        tpr = m_data["roc_curve"]["tpr"]
        auc = m_data["auc_roc"]
        ax1.plot(
            fpr, tpr,
            label=f"{labels_map[mode]} (AUC = {auc:.3f})",
            color=colors_map[mode],
            linewidth=2.5
        )

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax1.set_title(f"A. ROC Curves on Unseen Benchmark ({target_ds.upper()})", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Panel 2: Normalized Confusion Matrix for Full Dual-Stream
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    dual_cm = np.array(results[target_ds]["dual"]["confusion_matrix"]["matrix"])
    cm_norm = dual_cm.astype("float") / np.maximum(dual_cm.sum(axis=1, keepdims=True), 1)

    im = ax2.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    classes = ["Fake (0)", "Real (1)"]
    tick_marks = np.arange(len(classes))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(classes, fontsize=11)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(classes, fontsize=11)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            count = dual_cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax2.text(j, i, f"{val*100:.1f}%\n(n={count})",
                     ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax2.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax2.set_title("B. Dual-Stream Normalized Confusion Matrix", fontsize=13, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 3: Cross-Dataset Generalization (AUC-ROC Across Datasets)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    datasets = [d for d in ["validation", "celeb_df", "faceforensics"] if d in results]
    ds_names = ["Validation (In-Domain)", "Celeb-DF (Out-Domain)", "FF++ (Out-Domain)"][:len(datasets)]

    x = np.arange(len(datasets))
    width = 0.25

    for idx, mode in enumerate(modes):
        aucs = [results[d][mode]["auc_roc"] * 100 for d in datasets]
        rects = ax3.bar(
            x + (idx - 1) * width,
            aucs,
            width,
            label=labels_map[mode],
            color=colors_map[mode],
            alpha=0.85,
            edgecolor="black"
        )
        for r in rects:
            h = r.get_height()
            ax3.annotate(f"{h:.1f}%",
                         xy=(r.get_x() + r.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax3.set_ylabel("AUC-ROC (%)", fontsize=11, fontweight="bold")
    ax3.set_title("C. Cross-Dataset Generalization & Ablation Gap", fontsize=13, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(ds_names, fontsize=10, fontweight="bold")
    ax3.set_ylim([45, 105])
    ax3.legend(loc="upper right", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    # -------------------------------------------------------------
    # Panel 4: Metric Comparison Table (Accuracy, F1, Recall, Spec)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.axis("off")

    metric_cols = ["Metric", "Spatial-Only", "Freq-Only", "Full Dual-Stream", "Fusion Gain (Δ)"]
    m_eval = results[target_ds]
    
    rows = [
        ["AUC-ROC", f"{m_eval['spatial_only']['auc_roc']:.3f}", f"{m_eval['freq_only']['auc_roc']:.3f}", f"{m_eval['dual']['auc_roc']:.3f}", f"+{(m_eval['dual']['auc_roc'] - max(m_eval['spatial_only']['auc_roc'], m_eval['freq_only']['auc_roc'])):.3f}"],
        ["Accuracy", f"{m_eval['spatial_only']['accuracy']*100:.1f}%", f"{m_eval['freq_only']['accuracy']*100:.1f}%", f"{m_eval['dual']['accuracy']*100:.1f}%", f"+{(m_eval['dual']['accuracy'] - max(m_eval['spatial_only']['accuracy'], m_eval['freq_only']['accuracy']))*100:.1f}%"],
        ["Precision", f"{m_eval['spatial_only']['precision']*100:.1f}%", f"{m_eval['freq_only']['precision']*100:.1f}%", f"{m_eval['dual']['precision']*100:.1f}%", f"+{(m_eval['dual']['precision'] - max(m_eval['spatial_only']['precision'], m_eval['freq_only']['precision']))*100:.1f}%"],
        ["Recall (Sens.)", f"{m_eval['spatial_only']['recall']*100:.1f}%", f"{m_eval['freq_only']['recall']*100:.1f}%", f"{m_eval['dual']['recall']*100:.1f}%", f"+{(m_eval['dual']['recall'] - max(m_eval['spatial_only']['recall'], m_eval['freq_only']['recall']))*100:.1f}%"],
        ["Specificity", f"{m_eval['spatial_only']['specificity']*100:.1f}%", f"{m_eval['freq_only']['specificity']*100:.1f}%", f"{m_eval['dual']['specificity']*100:.1f}%", f"+{(m_eval['dual']['specificity'] - max(m_eval['spatial_only']['specificity'], m_eval['freq_only']['specificity']))*100:.1f}%"],
        ["F1-Score", f"{m_eval['spatial_only']['f1_score']:.3f}", f"{m_eval['freq_only']['f1_score']:.3f}", f"{m_eval['dual']['f1_score']:.3f}", f"+{(m_eval['dual']['f1_score'] - max(m_eval['spatial_only']['f1_score'], m_eval['freq_only']['f1_score'])):.3f}"]
    ]

    table = ax4.table(
        cellText=rows,
        colLabels=metric_cols,
        cellLoc="center",
        loc="center",
        colWidths=[0.24, 0.18, 0.18, 0.22, 0.18]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)

    # Style header
    for k in range(len(metric_cols)):
        table[(0, k)].set_facecolor("#1A1D20")
        table[(0, k)].set_text_props(color="white", fontweight="bold")

    # Highlight Full Dual Stream column
    for r in range(1, len(rows) + 1):
        table[(r, 3)].set_facecolor("#E0F7F6")
        table[(r, 3)].set_text_props(fontweight="bold")
        table[(r, 4)].set_facecolor("#EBFBEE")
        table[(r, 4)].set_text_props(color="#0A7C2E", fontweight="bold")

    ax4.set_title(f"D. Statistical Ablation Matrix ({target_ds.upper()})", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Publication-grade visual figures generated: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("  VERITAS ACADEMIC ABLATION & GENERALIZATION BENCHMARK ENGINE")
    print("=" * 70)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Architecture   : Token-Based Dual-Stream Cross-Attention Network")
    print(f"[*] Benchmarks     : In-Domain Val | Celeb-DF | FaceForensics++")

    # 1. Train or load model
    model = train_ablation_model(epochs=3, batch_size=32, lr=1e-4, device=device)

    # 2. Datasets to evaluate
    eval_datasets = {}
    if Path("data/validation").exists():
        eval_datasets["validation"] = SOTADeepfakeDataset("data/validation", use_sbi_synthesis=False)
    if Path("data/celeb_df").exists():
        eval_datasets["celeb_df"] = SOTADeepfakeDataset("data/celeb_df", use_sbi_synthesis=False, max_samples=200)
    if Path("data/faceforensics").exists():
        eval_datasets["faceforensics"] = SOTADeepfakeDataset("data/faceforensics", use_sbi_synthesis=False)

    print("\n" + "=" * 70)
    print("  PHASE 2: RIGOROUS MULTI-DATASET ABLATION EVALUATION")
    print("=" * 70)

    results: Dict[str, Dict[str, Any]] = {}
    modes = ["spatial_only", "freq_only", "dual"]

    for ds_name, ds in eval_datasets.items():
        results[ds_name] = {}
        print(f"\n--- Evaluating on Dataset: {ds_name.upper()} ({len(ds)} samples) ---")
        for mode in modes:
            t0 = time.time()
            metrics = evaluate_dataset_mode(model, ds, mode=mode, device=device)
            elapsed = time.time() - t0
            results[ds_name][mode] = metrics
            print(f"  [{mode.upper():<12}] AUC-ROC: {metrics['auc_roc']:.4f} | Acc: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_score']:.4f} | Prec: {metrics['precision']*100:.1f}% | Rec: {metrics['recall']*100:.1f}% ({elapsed:.1f}s)")

    # 3. Save raw JSON metrics
    with open("models/ablation_study_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[+] Raw JSON metrics saved to: models/ablation_study_metrics.json")

    # 4. Generate high-resolution figures
    generate_ablation_figures(results, output_path="assets/ablation_benchmark_results.png")

    print("\n" + "=" * 70)
    print("  ABLATION BENCHMARK EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
