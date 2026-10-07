"""
VERITAS SOTA Deepfake Cyber-Forensics Enterprise Server
High-Performance FastAPI REST Server & Static Asset Provider
Includes Multi-Modal Vision, Audio Vocoder, Generator Attribution,
Live Stream Interception, Case Management, SIEM Webhooks, and Prometheus Observability.
"""

import sys
import os
import io
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
from PIL import Image

import fastapi
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.face_detector import FaceForensicExtractor
from core.spatial_detector import SpatialForensicDetector
from core.frequency_forensics import FrequencyForensicAnalyzer
from core.ela_forensics import ELAForensicAnalyzer
from core.temporal_forensics import TemporalForensicAnalyzer
from core.explainability import GradCAMExplainer
from core.ensemble_engine import ForensicEnsembleEngine
from core.report_generator import ForensicReportGenerator
from core.audio_forensics import AudioForensicAnalyzer
from core.attribution_forensics import GeneratorAttributionEngine
from core.case_manager import CaseManager
from core.siem_dispatcher import SIEMAlertDispatcher


app = FastAPI(
    title="VERITAS Enterprise SOTA Forensic Suite",
    description="State-of-the-Art Deepfake Detection API, Audio Vocoder Forensics, Live KYC Defense & Case Vault",
    version="2.1.0-ENTERPRISE"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics & Observability Tracker
METRICS_DATA = {
    "total_inferences": 0,
    "synthetic_detections": 0,
    "start_time": time.time()
}

# Initialize singletons
print("[*] Initializing VERITAS Enterprise Forensic Engine Singletons...")
extractor = FaceForensicExtractor()
spatial_det = SpatialForensicDetector()
freq_analyzer = FrequencyForensicAnalyzer()
ela_analyzer = ELAForensicAnalyzer()
temp_analyzer = TemporalForensicAnalyzer()
engine = ForensicEnsembleEngine()
reporter = ForensicReportGenerator()
audio_analyzer = AudioForensicAnalyzer()
attribution_engine = GeneratorAttributionEngine()
case_manager = CaseManager()
siem_dispatcher = SIEMAlertDispatcher()
print("[+] All enterprise singletons initialized successfully.")


def ndarray_to_b64(img_rgb: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
    """Converts RGB numpy array into web-ready Base64 Data URL."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


# =====================================================================
# Telemetry, Health Probes & Prometheus Metrics
# =====================================================================

@app.get("/healthz")
async def health_check():
    """Kubernetes / Docker Liveness Probe."""
    return {"status": "HEALTHY", "service": "veritas-sota-forensics"}


@app.get("/readyz")
async def ready_check():
    """Readiness Probe: Confirms models are loaded into memory."""
    return {
        "status": "READY",
        "spatial_detector": spatial_det.model is not None,
        "device": str(spatial_det.device),
        "backbone": "Dual-Stream Cross-Attention Network"
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible plain text metrics endpoint."""
    uptime_s = time.time() - METRICS_DATA["start_time"]
    rate = METRICS_DATA["synthetic_detections"] / max(1, METRICS_DATA["total_inferences"])
    content = (
        f"# HELP veritas_inference_total Total deepfake forensic evaluations\n"
        f"# TYPE veritas_inference_total counter\n"
        f"veritas_inference_total {METRICS_DATA['total_inferences']}\n"
        f"# HELP veritas_synthetic_detections Total synthetic deepfakes flagged\n"
        f"# TYPE veritas_synthetic_detections counter\n"
        f"veritas_synthetic_detections {METRICS_DATA['synthetic_detections']}\n"
        f"# HELP veritas_synthetic_rate Ratio of flagged deepfakes\n"
        f"# TYPE veritas_synthetic_rate gauge\n"
        f"veritas_synthetic_rate {rate:.4f}\n"
        f"# HELP veritas_uptime_seconds Server uptime in seconds\n"
        f"# TYPE veritas_uptime_seconds gauge\n"
        f"veritas_uptime_seconds {uptime_s:.1f}\n"
    )
    return Response(content=content, media_type="text/plain; version=0.0.4")


@app.get("/api/telemetry")
async def get_telemetry():
    """Comprehensive system telemetry, accelerator, and model architecture specifications."""
    return {
        "status": "ONLINE",
        "system": "VERITAS Enterprise SOTA Forensics",
        "engine_version": "v2.1.0-ENTERPRISE",
        "device": str(spatial_det.device),
        "backbone": "Dual-Stream Cross-Attention Network (EfficientNet-B0 + SRM + FreqNet)",
        "modalities": [
            "Spatial Deep Feature Extraction",
            "2D Fourier & Azimuthal Radial Decay (1/f Profile)",
            "Error Level Analysis (JPEG Disparity & Boundary Seams)",
            "Biological Dynamics (Blink Frequency & Optical Flow Jitter)",
            "Audio Acoustic & Neural Vocoder Brick-Wall Analysis",
            "Multi-Generator Attribution (GAN vs Diffusion vs FaceSwap)",
            "Cryptographic Evidentiary Verification (SHA-256)"
        ],
        "siem_status": siem_dispatcher.get_status(),
        "total_cases": len(case_manager.list_cases())
    }


# =====================================================================
# Investigation Samples
# =====================================================================

@app.get("/api/samples")
async def list_samples():
    """List pre-packaged investigation assets."""
    assets_dir = Path("assets")
    samples = []
    if assets_dir.exists():
        for p in assets_dir.glob("*.jpg"):
            is_synthetic = "synthetic" in p.name.lower() or "fake" in p.name.lower()
            samples.append({
                "filename": p.name,
                "label": "AI Synthetic Sample" if is_synthetic else "Authentic Photographic Sample",
                "type": "synthetic" if is_synthetic else "authentic",
                "size_kb": round(p.stat().st_size / 1024, 1)
            })
    return {"samples": samples}


@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    """Retrieve sample file from assets."""
    sample_path = Path("assets") / filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(str(sample_path))


# =====================================================================
# Image Forensics & Generator Attribution
# =====================================================================

@app.post("/api/analyze/image")
async def analyze_image_endpoint(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None)
):
    """
    Main Multi-Modal Image Forensic Endpoint.
    Returns executive verdict, multi-cue scores, Grad-CAM, FFT spectrum, ELA heatmap,
    Generator Attribution, and SHA-256. Automatically dispatches alerts to SIEM.
    """
    raw_bytes = None
    media_name = "uploaded_media.jpg"

    if file is not None and file.filename:
        raw_bytes = await file.read()
        media_name = file.filename
    elif sample_name:
        sample_path = Path("assets") / sample_name
        if sample_path.exists():
            raw_bytes = sample_path.read_bytes()
            media_name = sample_name

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="No valid image or sample provided.")

    t0 = time.time()
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {e}")

    # 1. Face detection & extraction
    faces = extractor.detect_faces(img_rgb)
    primary_face = faces[0]
    face_crop = primary_face["face_crop"]
    face_box = primary_face["box"]

    # 2. Multi-domain forensic evaluation
    spatial_res = spatial_det.predict(face_crop)
    freq_res = freq_analyzer.analyze(face_crop)
    ela_res = ela_analyzer.analyze(img_rgb, face_box)

    # 3. Bayesian multi-cue fusion
    fused = engine.fuse_image_evidence(spatial_res, freq_res, ela_res, raw_bytes)

    # 4. Generator Attribution (GAN, Diffusion, FaceSwap, Camera)
    attribution_res = attribution_engine.attribute(fused)

    # 5. Grad-CAM XAI explanation
    explainer = GradCAMExplainer(spatial_det)
    overlay_cam, raw_cam, cam_meta = explainer.generate_heatmap(face_crop, target_class=0)
    explainer.remove_hooks()

    # 6. Cyberpunk HUD overlay
    hud_overlay = extractor.draw_forensic_hud(img_rgb, faces, fused["fused_fake_prob"])

    elapsed_ms = (time.time() - t0) * 1000.0

    # Observability metric update
    METRICS_DATA["total_inferences"] += 1
    if fused["fused_fake_prob"] >= 0.50:
        METRICS_DATA["synthetic_detections"] += 1

    response_payload = {
        "media_name": media_name,
        "elapsed_ms": round(elapsed_ms, 1),
        "faces_detected": len(faces),
        "is_fallback": primary_face.get("is_fallback", False),
        "verdict": fused["verdict"],
        "severity": fused["severity"],
        "badge_color": fused["badge_color"],
        "confidence": fused["confidence"],
        "fused_fake_prob": fused["fused_fake_prob"],
        "fused_real_prob": fused["fused_real_prob"],
        "sha256": fused["sha256"],
        "cue_breakdown": fused["cue_breakdown"],
        "anomalies": fused["anomalies"],
        "generator_attribution": attribution_res,
        "spectral_metrics": {
            "slope": freq_res["spectral_slope"],
            "hfer": freq_res["hfer"],
            "residual": freq_res["spectral_residual"],
            "radial_profile": freq_res["radial_profile"]
        },
        "ela_metrics": {
            "disparity_ratio": ela_res["disparity_ratio"],
            "face_mean": ela_res["face_ela_mean"],
            "bg_mean": ela_res["background_ela_mean"]
        },
        "cam_meta": cam_meta,
        "images": {
            "hud": ndarray_to_b64(hud_overlay),
            "face_crop": ndarray_to_b64(face_crop),
            "gradcam_overlay": ndarray_to_b64(overlay_cam),
            "gradcam_raw": ndarray_to_b64(raw_cam),
            "fft_spectrum": ndarray_to_b64(freq_res["spectrum_image"]),
            "ela_heatmap": ndarray_to_b64(ela_res["ela_heatmap"]),
            "ela_visual": ndarray_to_b64(ela_res["ela_visual"])
        }
    }

    # Asynchronous SIEM incident dispatch
    try:
        siem_dispatcher.dispatch_incident_alert(response_payload)
    except Exception as e:
        print(f"[!] SIEM dispatch error: {e}")

    return response_payload


# =====================================================================
# Video Temporal Stream Forensics
# =====================================================================

@app.post("/api/analyze/video")
async def analyze_video_endpoint(
    file: UploadFile = File(...)
):
    """
    Video Temporal Stream Forensics.
    Extracts frames, analyzes biological blink dynamics & optical flow motion jitter.
    """
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="Unable to decode video stream.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // 30)

    frame_records = []
    frame_idx = 0
    sampled_faces = []

    while cap.isOpened() and len(frame_records) < 30:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = extractor.detect_faces(frame_rgb)
            face_crop = faces[0]["face_crop"]
            spatial_res = spatial_det.predict(face_crop)
            frame_records.append({
                "frame_idx": frame_idx,
                "timestamp": round(frame_idx / fps, 2),
                "fake_prob": spatial_res["fake_probability"],
                "confidence": spatial_res["confidence"]
            })
            sampled_faces.append(face_crop)
        frame_idx += 1

    cap.release()
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    temp_res = temp_analyzer.analyze_sequence(sampled_faces, fps=float(fps / sample_rate))
    v_verdict = engine.fuse_video_sequence(frame_records, temp_res, content)

    # Observability metric update
    METRICS_DATA["total_inferences"] += 1
    if v_verdict["fused_fake_prob"] >= 0.50:
        METRICS_DATA["synthetic_detections"] += 1

    video_response = {
        "media_name": file.filename,
        "verdict": v_verdict["verdict"],
        "severity": v_verdict["severity"],
        "badge_color": v_verdict["badge_color"],
        "confidence": v_verdict["confidence"],
        "fused_fake_prob": v_verdict["fused_fake_prob"],
        "total_frames_analyzed": v_verdict["total_frames_analyzed"],
        "fake_frame_count": v_verdict["fake_frame_count"],
        "fake_frame_ratio": v_verdict["fake_frame_ratio"],
        "sha256": v_verdict["sha256"],
        "anomalies": v_verdict["anomalies"],
        "temporal_metrics": temp_res,
        "timeline": [
            {
                "frame": f["frame_idx"],
                "timestamp": f["timestamp"],
                "fake_prob": round(f["fake_prob"] * 100, 1),
                "confidence": round(f["confidence"], 1)
            }
            for f in frame_records
        ]
    }

    try:
        siem_dispatcher.dispatch_incident_alert(video_response)
    except Exception:
        pass

    return video_response


# =====================================================================
# Audio Acoustic & Neural Vocoder Forensics
# =====================================================================

@app.post("/api/analyze/audio")
async def analyze_audio_endpoint(
    file: UploadFile = File(...)
):
    """
    Acoustic & Neural Vocoder Audio Forensics.
    Detects synthetic voice cloning (ElevenLabs, Tortoise, RVC, VITS) via high-band
    brick-wall cutoff, pitch jitter micro-tremors, and STFT spectrograms.
    """
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded.")
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    t0 = time.time()
    result = audio_analyzer.analyze(raw_bytes)
    result["media_name"] = file.filename
    result["elapsed_ms"] = round((time.time() - t0) * 1000.0, 1)

    # Observability metric update
    METRICS_DATA["total_inferences"] += 1
    if result["synthetic_probability"] >= 0.50:
        METRICS_DATA["synthetic_detections"] += 1

    try:
        siem_dispatcher.dispatch_incident_alert(result)
    except Exception:
        pass

    return result


# =====================================================================
# Live Stream & Real-Time KYC Webcam Interceptor
# =====================================================================

class StreamFramePayload(BaseModel):
    frame_b64: str  # Base64 JPEG data URL


@app.post("/api/analyze/stream-frame")
async def analyze_stream_frame(payload: StreamFramePayload):
    """
    Ultra-low latency frame interceptor for real-time KYC, video conferencing,
    and anti-spoofing impersonation defense. Returns analysis in <100ms.
    """
    try:
        data_str = payload.frame_b64
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(data_str)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        frame_rgb = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode failure: {e}")

    t0 = time.time()
    faces = extractor.detect_faces(frame_rgb)
    if not faces or faces[0].get("is_fallback", False):
        return {
            "face_detected": False,
            "threat_status": "NO_SUBJECT_FOUND",
            "fake_prob": 0.0,
            "liveness_score": 0.0,
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }

    face = faces[0]
    box = face["box"]  # [x, y, w, h]
    face_crop = face["face_crop"]

    # Rapid inference
    spatial_res = spatial_det.predict(face_crop)
    fake_prob = float(spatial_res["fake_probability"])

    # Liveness check: high-frequency variance of Laplacian
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    liveness_score = round(min(99.0, max(15.0, lap_var / 4.5)), 1)

    if fake_prob >= 0.70:
        threat_status = "SYNTHETIC_AVATAR_ALERT"
        badge_color = "#ef4444"
        verdict = "SYNTHETIC"
    elif fake_prob >= 0.40:
        threat_status = "SUSPICIOUS_ARTIFACTS"
        badge_color = "#f59e0b"
        verdict = "SUSPICIOUS"
    else:
        threat_status = "AUTHENTIC_LIVE_SUBJECT"
        badge_color = "#10b981"
        verdict = "AUTHENTIC"

    latency_ms = round((time.time() - t0) * 1000.0, 1)

    return {
        "face_detected": True,
        "box": [int(b) for b in box],
        "threat_status": threat_status,
        "verdict": verdict,
        "badge_color": badge_color,
        "fake_probability": round(fake_prob * 100.0, 1),
        "liveness_score": liveness_score,
        "latency_ms": latency_ms
    }


# =====================================================================
# Investigative Case Management & Evidentiary Master Dossiers
# =====================================================================

class CreateCasePayload(BaseModel):
    title: str
    investigator: str
    organization: str


class AttachExhibitPayload(BaseModel):
    media_name: str
    verdict: str
    confidence: float
    sha256: str
    threat_actor: Optional[str] = "Unknown"
    notes: Optional[str] = ""


@app.get("/api/cases")
async def list_cases_endpoint():
    """List all open and archived investigative cases."""
    return {"cases": case_manager.list_cases()}


@app.post("/api/cases")
async def create_case_endpoint(payload: CreateCasePayload):
    """Open a new investigative case."""
    new_case = case_manager.create_case(payload.title, payload.investigator, payload.organization)
    return new_case


@app.get("/api/cases/{case_id}")
async def get_case_endpoint(case_id: str):
    """Retrieve full details and exhibit inventory for a case."""
    cases = {c["case_id"]: c for c in case_manager.list_cases()}
    if case_id not in cases:
        raise HTTPException(status_code=404, detail="Case not found")
    return cases[case_id]


@app.post("/api/cases/{case_id}/exhibits")
async def attach_exhibit_endpoint(case_id: str, exhibit: AttachExhibitPayload):
    """Attach an analyzed forensic exhibit to a case."""
    updated = case_manager.add_exhibit(case_id, exhibit.dict())
    if not updated:
        raise HTTPException(status_code=404, detail="Case not found")
    return updated


@app.get("/api/cases/{case_id}/export/pdf")
async def export_case_pdf_endpoint(case_id: str):
    """Export comprehensive Judicial Master Evidence Dossier PDF."""
    pdf_bytes = case_manager.generate_case_master_pdf(case_id)
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="Case dossier generation failed")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=VERITAS_Master_Dossier_{case_id}.pdf"}
    )


# =====================================================================
# SIEM & SOC Alert Integration
# =====================================================================

class SIEMConfigPayload(BaseModel):
    enabled: bool
    webhook_url: str
    alert_threshold: float
    hmac_secret: Optional[str] = ""


@app.get("/api/siem/config")
async def get_siem_config_endpoint():
    """Retrieve current SIEM integration configuration and dispatch history."""
    return siem_dispatcher.get_status()


@app.post("/api/siem/config")
async def update_siem_config_endpoint(payload: SIEMConfigPayload):
    """Update SIEM webhook endpoint, alert threshold, and HMAC secret."""
    return siem_dispatcher.update_config(
        payload.enabled,
        payload.webhook_url,
        payload.alert_threshold,
        payload.hmac_secret or ""
    )


@app.post("/api/siem/test")
async def test_siem_ping_endpoint():
    """Send canary ping to test webhook connectivity."""
    return siem_dispatcher.test_ping()


# =====================================================================
# Single Exhibit Export (PDF & JSON)
# =====================================================================

class ReportPayload(BaseModel):
    media_name: str
    forensic_result: dict


@app.post("/api/export/pdf")
async def export_pdf(payload: ReportPayload):
    """Generate and return official PDF forensic inspection certificate."""
    pdf_bytes = reporter.generate_pdf_report(payload.media_name, payload.forensic_result)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=VERITAS_Report_{payload.media_name}.pdf"}
    )


@app.post("/api/export/json")
async def export_json(payload: ReportPayload):
    """Generate and return JSON evidence dossier."""
    json_str = reporter.generate_json_report(payload.media_name, payload.forensic_result)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=VERITAS_Dossier_{payload.media_name}.json"}
    )


@app.get("/api/ablation-metrics")
async def get_ablation_metrics():
    """Returns empirical academic ablation metrics across spatial, frequency, and dual-stream architectures."""
    metrics_path = Path("models/ablation_study_metrics.json")
    if not metrics_path.exists():
        return JSONResponse({"status": "pending", "message": "Ablation evaluation in progress or not yet run."})
    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse({"status": "success", "benchmarks": data})



# =====================================================================
# Static Web Application Mount
# =====================================================================

web_dir = Path("web")
if web_dir.exists():
    app.mount("/static", StaticFiles(directory="web"), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse("web/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
