# 🚀 VERITAS Quick Start Guide

**VERITAS** is an Enterprise SOTA Multi-Modal Deepfake Defense & Evidentiary Cyber-Forensics Suite.

---

## 📋 Prerequisites

- **Python**: Version 3.10+ (Python 3.11 recommended)
- **Dependencies**: Install via `pip install -r requirements.txt`
- **Models**: Pre-trained weights located in `models/` (`sota_model.pth`, `best_model.pth`)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify model weights (auto-creates baseline if absent)
python init_weights.py
```

---

## 1. 🌐 Cyber-Forensics Web Studio

The Web Studio provides a real-time, interactive HUD featuring 9 forensic inspection laboratories:
- Image Spatial-Frequency Analysis with interactive Grad-CAM heatmap overlay
- Video Temporal Consistency & Blink Rate Timeline
- Neural Vocoder Audio Forensics & Spectrogram Player
- Live Real-Time KYC Webcam Interceptor
- Synthesizer Attribution Radar (GAN vs Diffusion vs FaceSwap vs Authentic)
- Chain-of-Custody Case Vault with Judicial Master PDF Dossier Export

### Launch the Studio

```bash
# Option A: 1-Click Launcher (Windows)
.\run_website.bat

# Option B: Direct Python Launch
python server.py
```

Open your browser at **[http://localhost:8000](http://localhost:8000)**.  
Interactive REST API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 2. 🛠️ Command-Line Interface (CLI)

The VERITAS CLI supports automated single-file, temporal video, and batch e-discovery auditing.

### Analyze a Single Image
```bash
python inference_cli.py --image path/to/image.jpg
```

### Export Official Audit Reports (JSON & PDF)
```bash
python inference_cli.py --image path/to/image.jpg --output-pdf report.pdf --output-json audit.json
```

### Analyze a Video Stream
Process every 5th frame with biological eye-blink tracking and optical flow consistency:
```bash
python inference_cli.py --video path/to/video.mp4 --frame-skip 5
```

### Batch Audit an Entire Directory
```bash
python inference_cli.py --batch path/to/evidence_folder/
```

---

## 3. 🐍 Enterprise Python SDK

Integrate VERITAS into automated detection pipelines, SOC tools, or KYC workflows:

```python
from veritas_sdk import VeritasClient

# Initialize client pointing to local or remote VERITAS engine
client = VeritasClient("http://localhost:8000")

# 1. Analyze image media
image_result = client.analyze_image("evidence.jpg")
print(f"Verdict: {image_result.verdict} ({image_result.fake_probability * 100:.1f}%)")
print(f"Attributed Synthesizer: {image_result.primary_generator}")
print(f"SHA-256 Hash: {image_result.sha256}")

# 2. Analyze voice audio clip
audio_result = client.analyze_audio("voice_call.wav")
print(f"Audio Verdict: {audio_result.verdict} ({audio_result.synthetic_probability * 100:.1f}%)")

# 3. Create legal investigation case and attach exhibits
case = client.create_case(
    title="Executive Impersonation Case",
    investigator="Senior Security Analyst",
    organization="Cyber Forensics Unit"
)
client.attach_evidence(case.case_id, image_result)
client.attach_evidence(case.case_id, audio_result)

# 4. Download Certified Multi-Exhibit Master PDF Dossier
client.download_case_pdf(case.case_id, "Master_Case_Dossier.pdf")
```

---

## 4. 🧠 SOTA Training & Ablation Engine

VERITAS includes a complete training pipeline with dynamic Self-Blended Images (SBI) and dual-stream cross-attention architectures.

```bash
# Run training with Self-Blended Images (SBI)
python training/train_sota.py --epochs 15 --batch-size 16 --lr 0.0001

# Execute benchmark ablation evaluations
python training/train_and_evaluate_ablation.py
```

---

## 5. 📊 Output Interpretation

```
==============================================================
VERITAS CYBER-FORENSIC AUDIT
==============================================================
Target: evidence.jpg
SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
--------------------------------------------------------------
Verdict:               🔴 CRITICAL: SYNTHETIC DEEPFAKE DETECTED
Forensic Confidence:   98.40%
Primary Synthesizer:   Latent Diffusion (Midjourney/Flux/SD) (88.5%)
Severity Level:        CRITICAL
--------------------------------------------------------------
Multi-Domain Evidence:
  • Spatial Residuals:      94.2% Anomaly
  • Spectral 2D-FFT Grid:   97.8% Anomaly
  • ELA Compression Seam:   91.5% Disparity
==============================================================
```
