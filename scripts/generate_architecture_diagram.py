"""
Generates publication-quality neural network architecture block diagram for the
VERITAS Dual-Stream Spatial-Frequency Cross-Attention Deepfake Detector.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture():
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Header Title
    ax.text(9, 9.4, "VERITAS: Dual-Stream Spatial-Frequency Cross-Attention Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
    ax.text(9, 8.95, "Mathematical Fusion of Spatial Visual Tokens and 2D-FFT Spectral Frequency Representations",
            ha="center", va="center", fontsize=11.5, color="#475569", style="italic")

    def draw_box(x, y, w, h, title, subtitle="", color="#F8FAFC", border="#64748B", title_col="#0F172A", title_size=10, sub_size=8.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                      facecolor=color, edgecolor=border, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_col, zorder=4)
            ax.text(x + w/2, y + h*0.35, subtitle, ha="center", va="center", fontsize=sub_size, color="#475569", zorder=4)
        else:
            ax.text(x + w/2, y + h*0.5, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_col, zorder=4)

    def draw_arrow(x1, y1, x2, y2, label="", rad=0.0, lw=1.8, color="#1E293B"):
        connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14, connectionstyle=connectionstyle), zorder=5)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.16, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color="#334155",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="none", alpha=0.9), zorder=6)

    # 1. INPUT IMAGE (Left)
    draw_box(0.5, 3.8, 1.8, 2.2, "Input Face Crop", "RGB Tensor\n(B, 3, 224, 224)", color="#F1F5F9", border="#475569", title_size=10.5)

    # Branching arrows
    draw_arrow(2.3, 5.3, 3.1, 6.7, label="RGB", rad=0.1)
    draw_arrow(2.3, 4.5, 3.1, 3.1, label="RGB", rad=-0.1)

    # -------------------------------------------------------------
    # STREAM 1: SPATIAL VISUAL STREAM (Top)
    # -------------------------------------------------------------
    stream1 = patches.FancyBboxPatch((3.0, 5.5), 5.7, 2.7, boxstyle="round,pad=0.12,rounding_size=0.2",
                                    facecolor="#F5F3FF", edgecolor="#8B5CF6", linestyle="--", linewidth=1.5, zorder=1)
    ax.add_patch(stream1)
    ax.text(3.3, 7.9, "Stream 1: Spatial Visual Stream", fontsize=11, fontweight="bold", color="#6D28D9")

    draw_box(3.2, 5.8, 2.2, 1.8, "SRM Conv Bank", "Fixed High-Pass\nResidual Kernels", color="#EDE9FE", border="#8B5CF6", title_col="#5B21B6")
    draw_arrow(5.4, 6.7, 5.9, 6.7)
    draw_box(5.9, 5.8, 2.6, 1.8, "EfficientNet-B0", "Convolutional Features\nF_s in (B, 1280, 7, 7)", color="#EDE9FE", border="#8B5CF6", title_col="#5B21B6")

    # -------------------------------------------------------------
    # STREAM 2: 2D-FFT SPECTRAL FREQUENCY STREAM (Bottom)
    # -------------------------------------------------------------
    stream2 = patches.FancyBboxPatch((3.0, 1.6), 5.7, 2.7, boxstyle="round,pad=0.12,rounding_size=0.2",
                                    facecolor="#FDF2F8", edgecolor="#EC4899", linestyle="--", linewidth=1.5, zorder=1)
    ax.add_patch(stream2)
    ax.text(3.3, 4.0, "Stream 2: 2D-FFT Spectral Stream", fontsize=11, fontweight="bold", color="#BE185D")

    draw_box(3.2, 1.9, 2.2, 1.8, "2D Fast Fourier\nTransform (FFT)", "Centered Log-Magnitude\nF_fft in (B, 3, 224, 224)", color="#FCE7F3", border="#EC4899", title_col="#9D174D")
    draw_arrow(5.4, 2.8, 5.9, 2.8)
    draw_box(5.9, 1.9, 2.6, 1.8, "Spectral ConvNet", "5-Stage Residual Conv\nF_f in (B, 256, 7, 7)", color="#FCE7F3", border="#EC4899", title_col="#9D174D")

    # -------------------------------------------------------------
    # TOKEN RESHAPING & POSITIONAL ENCODINGS
    # -------------------------------------------------------------
    draw_arrow(8.5, 6.7, 9.3, 6.7, label="Flatten (B, 49, 1280)")
    draw_arrow(8.5, 2.8, 9.3, 2.8, label="Flatten (B, 49, 256)")

    draw_box(9.3, 5.8, 2.3, 1.8, "Spatial Tokens", "Linear Proj + PosEmbed\nT_s in (B, 49, 256)", color="#EDE9FE", border="#8B5CF6", title_col="#5B21B6")
    draw_box(9.3, 1.9, 2.3, 1.8, "Spectral Tokens", "Linear Proj + PosEmbed\nT_f in (B, 49, 256)", color="#FCE7F3", border="#EC4899", title_col="#9D174D")

    # Connect tokens to Cross-Attention Module
    draw_arrow(11.6, 6.7, 12.4, 5.7, label="Keys & Values (K, V)", rad=-0.08)
    draw_arrow(11.6, 2.8, 12.4, 4.3, label="Queries (Q)", rad=0.08)

    # -------------------------------------------------------------
    # MULTI-HEAD CROSS-ATTENTION FUSION (Center-Right)
    # -------------------------------------------------------------
    fusion_box = patches.FancyBboxPatch((12.4, 2.4), 2.8, 5.2, boxstyle="round,pad=0.15,rounding_size=0.2",
                                       facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.8, zorder=1)
    ax.add_patch(fusion_box)
    ax.text(13.8, 7.3, "Multi-Head Cross-Attention", ha="center", fontsize=11, fontweight="bold", color="#065F46")
    ax.text(13.8, 6.95, "8 Heads | N = 49 Tokens | D = 256", ha="center", fontsize=8.5, color="#047857")

    draw_box(12.6, 5.5, 2.4, 1.1, "Scaled Dot-Product", "Attn = Softmax(Q K^T / sqrt(D))\nMatrix: (B, 49, 49)", color="#D1FAE5", border="#10B981", title_col="#065F46", title_size=9)
    draw_arrow(13.8, 5.5, 13.8, 4.9)
    draw_box(12.6, 3.8, 2.4, 1.1, "Value Aggregation", "Attended = Attn x V\nResidual + LayerNorm", color="#D1FAE5", border="#10B981", title_col="#065F46", title_size=9)
    draw_arrow(13.8, 3.8, 13.8, 3.4)
    draw_box(12.6, 2.6, 2.4, 0.8, "FeedForward MLP", "GELU + LayerNorm", color="#A7F3D0", border="#059669", title_col="#065F46", title_size=9)

    # -------------------------------------------------------------
    # CLASSIFICATION HEAD & VERDICT (Right)
    # -------------------------------------------------------------
    draw_arrow(15.2, 5.0, 15.8, 5.0, label="Fused (512-dim)")

    draw_box(15.8, 3.6, 1.8, 2.8, "Classifier Head", "Global AvgPool\nLinear(512, 256)\nBatchNorm + GELU\nLinear(256, 2)\nSoftmax", color="#EFF6FF", border="#3B82F6", title_col="#1E40AF", title_size=10)

    # Verdict arrows
    ax.annotate("", xy=(17.8, 5.6), xytext=(17.6, 5.3), arrowprops=dict(arrowstyle="-|>", color="#10B981", lw=2.5))
    ax.annotate("", xy=(17.8, 4.4), xytext=(17.6, 4.7), arrowprops=dict(arrowstyle="-|>", color="#EF4444", lw=2.5))

    ax.text(17.85, 5.6, "Authentic (P_real)", fontsize=9.5, fontweight="bold", color="#059669", va="center")
    ax.text(17.85, 4.4, "Deepfake (P_fake)", fontsize=9.5, fontweight="bold", color="#DC2626", va="center")

    # Bottom Caption / Formula
    ax.text(9, 0.8, "Cross-Attention Mechanism: Q in R^(49 x 256) (Spectral Query), K, V in R^(49 x 256) (Spatial Keys/Values). Enables spectral anomalies to dynamically attend to localized facial features.",
            ha="center", va="center", fontsize=9.5, color="#475569", style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.9))

    plt.tight_layout()
    Path("assets").mkdir(exist_ok=True)
    out_path = Path("assets/architecture_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Architecture diagram re-rendered: {out_path}")

if __name__ == "__main__":
    draw_architecture()
