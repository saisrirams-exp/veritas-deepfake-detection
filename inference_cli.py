#!/usr/bin/env python3
"""
SOTA Deepfake Detection & Forensic CLI Suite
Multi-Domain Analysis: Spatial Deep Learning + 2D-FFT + Error Level Analysis (ELA) + Temporal Dynamics
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.face_detector import FaceForensicExtractor
from core.spatial_detector import SpatialForensicDetector
from core.frequency_forensics import FrequencyForensicAnalyzer
from core.ela_forensics import ELAForensicAnalyzer
from core.temporal_forensics import TemporalForensicAnalyzer
from core.ensemble_engine import ForensicEnsembleEngine
from core.report_generator import ForensicReportGenerator


def analyze_single_image(
    image_path: Path,
    extractor: FaceForensicExtractor,
    spatial_det: SpatialForensicDetector,
    freq_analyzer: FrequencyForensicAnalyzer,
    ela_analyzer: ELAForensicAnalyzer,
    engine: ForensicEnsembleEngine,
    verbose: bool = True
):
    """Run full multi-domain forensic pipeline on an image."""
    if not image_path.exists():
        print(f"[!] File not found: {image_path}", file=sys.stderr)
        return None

    raw_bytes = image_path.read_bytes()
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"[!] Unable to decode image: {image_path}", file=sys.stderr)
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Detect faces
    faces = extractor.detect_faces(img_rgb)
    primary_face = faces[0]
    face_crop = primary_face["face_crop"]
    face_box = primary_face["box"]

    # 2. Multi-domain evaluations
    spatial_res = spatial_det.predict(face_crop)
    freq_res = freq_analyzer.analyze(face_crop)
    ela_res = ela_analyzer.analyze(img_rgb, face_box)

    # 3. Fuse evidence
    fused_res = engine.fuse_image_evidence(spatial_res, freq_res, ela_res, raw_bytes)
    fused_res["target_file"] = str(image_path)
    fused_res["faces_detected"] = len(faces)
    fused_res["is_fallback"] = primary_face.get("is_fallback", False)

    if verbose:
        print("\n" + "=" * 68)
        print(f"  🔬 SOTA FORENSIC DOSSIER: {image_path.name}")
        print("=" * 68)
        print(f"  SHA-256 Hash   : {fused_res['sha256']}")
        print(f"  Faces Detected : {len(faces)} {'(Fallback Full Frame)' if fused_res['is_fallback'] else ''}")
        print("-" * 68)
        
        # Colorized verdict
        v = fused_res['verdict']
        print(f"  PRIMARY VERDICT: \033[1m{v}\033[0m")
        print(f"  THREAT LEVEL   : {fused_res['severity']} | CONFIDENCE: {fused_res['confidence']:.2f}%")
        print(f"  SYNTHETIC PROB : {fused_res['fused_fake_prob']*100:.2f}% | AUTHENTIC PROB: {fused_res['fused_real_prob']*100:.2f}%")
        print("-" * 68)
        print("  MULTI-DOMAIN EVIDENCE BREAKDOWN:")
        for cue, score in fused_res['cue_breakdown'].items():
            bar_len = int(score / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    • {cue:<25}: [{bar}] {score:>5.1f}% Synthetic")

        print("-" * 68)
        print("  KEY FORENSIC OBSERVATIONS:")
        for anom in fused_res['anomalies']:
            print(f"    ⚠ {anom}")
        print("=" * 68 + "\n")

    return fused_res


def analyze_video(
    video_path: Path,
    extractor: FaceForensicExtractor,
    spatial_det: SpatialForensicDetector,
    freq_analyzer: FrequencyForensicAnalyzer,
    ela_analyzer: ELAForensicAnalyzer,
    temp_analyzer: TemporalForensicAnalyzer,
    engine: ForensicEnsembleEngine,
    frame_skip: int = 5,
    verbose: bool = True
):
    """Run temporal and multi-cue analysis on a video file."""
    if not video_path.exists():
        print(f"[!] File not found: {video_path}", file=sys.stderr)
        return None

    raw_bytes = video_path.read_bytes()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Unable to open video: {video_path}", file=sys.stderr)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    frame_records = []
    frames_for_temporal = []

    if verbose:
        print(f"\n[*] Scanning video stream: {video_path.name} (Sampling every {frame_skip} frames)...")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = extractor.detect_faces(frame_rgb)
            if faces:
                pf = faces[0]
                fc = pf["face_crop"]
                fb = pf["box"]
                sp = spatial_det.predict(fc)
                fr = freq_analyzer.analyze(fc)
                el = ela_analyzer.analyze(frame_rgb, fb)
                fused = engine.fuse_image_evidence(sp, fr, el)

                frame_records.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(frame_idx / fps, 2),
                    "fused_result": fused,
                    "fake_prob": fused["fused_fake_prob"]
                })
                frames_for_temporal.append({
                    "frame_idx": frame_idx,
                    "face_crop": fc,
                    "eyes": pf.get("eyes", []),
                    "spatial_fake_prob": fused["fused_fake_prob"]
                })

        frame_idx += 1

    cap.release()

    if not frame_records:
        print("[!] No valid frames or faces detected in video stream.")
        return None

    # Temporal analysis
    temp_res = temp_analyzer.analyze_frame_sequence(frames_for_temporal, fps=fps)
    video_verdict = engine.fuse_video_evidence(frame_records, temp_res, raw_bytes)
    video_verdict["target_file"] = str(video_path)

    if verbose:
        print("\n" + "=" * 68)
        print(f"  🎬 VIDEO FORENSIC DOSSIER: {video_path.name}")
        print("=" * 68)
        print(f"  SHA-256 Hash       : {video_verdict['sha256']}")
        print(f"  Frames Sampled     : {video_verdict['total_frames_analyzed']}")
        print(f"  Estimated Blinks   : {temp_res['blinks_detected']} ({temp_res['blink_rate_per_min']} blinks/min)")
        print(f"  Flow Jitter Score  : {temp_res['optical_flow_jitter']}")
        print("-" * 68)
        print(f"  PRIMARY VERDICT    : \033[1m{video_verdict['verdict']}\033[0m")
        print(f"  THREAT SEVERITY    : {video_verdict['severity']} | CONFIDENCE: {video_verdict['confidence']:.2f}%")
        print(f"  SUSPECT FRAMES     : {video_verdict['fake_frame_count']} / {video_verdict['total_frames_analyzed']} ({video_verdict['fake_frame_ratio']}%)")
        print("-" * 68)
        print("  KEY FORENSIC FINDINGS:")
        for anom in video_verdict['anomalies']:
            print(f"    ⚠ {anom}")
        print("=" * 68 + "\n")

    return video_verdict


def main():
    parser = argparse.ArgumentParser(description="SOTA Deepfake Detection & Forensic CLI Suite")
    parser.add_argument("--image", type=str, help="Path to input image for single-frame forensic analysis")
    parser.add_argument("--video", type=str, help="Path to input video for temporal stream forensic analysis")
    parser.add_argument("--batch", type=str, help="Path to directory containing images/videos to batch process")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to custom model weights")
    parser.add_argument("--frame-skip", type=int, default=5, help="Frame sampling interval for video streams")
    parser.add_argument("--output-json", type=str, help="Save forensic audit report to JSON file")
    parser.add_argument("--output-pdf", type=str, help="Save forensic audit certificate to PDF file")
    args = parser.parse_args()

    if not (args.image or args.video or args.batch):
        parser.print_help()
        sys.exit(1)

    print("[*] Initializing SOTA Forensic Engine modules...")
    extractor = FaceForensicExtractor()
    spatial_det = SpatialForensicDetector(model_path=args.model)
    freq_analyzer = FrequencyForensicAnalyzer()
    ela_analyzer = ELAForensicAnalyzer()
    temp_analyzer = TemporalForensicAnalyzer()
    engine = ForensicEnsembleEngine()
    reporter = ForensicReportGenerator()

    final_result = None

    if args.image:
        final_result = analyze_single_image(
            Path(args.image), extractor, spatial_det, freq_analyzer, ela_analyzer, engine
        )
    elif args.video:
        final_result = analyze_video(
            Path(args.video), extractor, spatial_det, freq_analyzer, ela_analyzer, temp_analyzer, engine,
            frame_skip=args.frame_skip
        )
    elif args.batch:
        batch_dir = Path(args.batch)
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [f for f in batch_dir.iterdir() if f.suffix.lower() in image_exts]
        print(f"[*] Found {len(files)} media files in {batch_dir}...")
        batch_results = []
        for f in files:
            res = analyze_single_image(
                f, extractor, spatial_det, freq_analyzer, ela_analyzer, engine, verbose=False
            )
            if res:
                batch_results.append(res)
                v_short = "FAKE" if "DEEPFAKE" in res['verdict'] else "REAL"
                print(f"  • {f.name:<30} -> {v_short:<6} (Confidence: {res['confidence']:.1f}%)")

        final_result = {"batch_summary": batch_results, "total_scanned": len(batch_results)}

    # Export options
    if final_result and args.output_json:
        json_content = reporter.generate_json_report(
            args.image or args.video or args.batch, final_result
        )
        Path(args.output_json).write_text(json_content, encoding="utf-8")
        print(f"[+] Forensic JSON report written to: {args.output_json}")

    if final_result and args.output_pdf:
        pdf_bytes = reporter.generate_pdf_report(
            args.image or args.video or args.batch, final_result
        )
        Path(args.output_pdf).write_bytes(pdf_bytes)
        print(f"[+] Forensic PDF certificate written to: {args.output_pdf}")

    # Return exit code based on verdict
    if final_result and "fused_fake_prob" in final_result:
        if final_result["fused_fake_prob"] >= 0.60:
            sys.exit(1)  # Deepfake detected
        elif final_result["fused_fake_prob"] >= 0.35:
            sys.exit(2)  # Suspicious / Inconclusive
    sys.exit(0)


if __name__ == "__main__":
    main()
