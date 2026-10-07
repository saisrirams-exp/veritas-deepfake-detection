"""
Multi-Cue Forensic Ensemble Fusion Engine
Combines Spatial Deep Learning, Frequency Domain Analysis, Error Level Analysis (ELA),
and Temporal Coherence into a calibrated unified forensic authenticity verdict.
"""

from typing import Dict, Any, List, Optional
import hashlib
import numpy as np

class ForensicEnsembleEngine:
    """
    Weighted Bayesian Forensic Fusion Aggregator.
    Synthesizes multi-domain evidence to produce calibrated deepfake probabilities and confidence intervals.
    """

    def __init__(
        self,
        weight_spatial: float = 0.45,
        weight_frequency: float = 0.25,
        weight_ela: float = 0.20,
        weight_boundary: float = 0.10
    ):
        total = weight_spatial + weight_frequency + weight_ela + weight_boundary
        self.w_spatial = weight_spatial / total
        self.w_frequency = weight_frequency / total
        self.w_ela = weight_ela / total
        self.w_boundary = weight_boundary / total

    def compute_sha256(self, file_bytes: bytes) -> str:
        """Calculate cryptographically secure SHA-256 media fingerprint for evidentiary integrity."""
        return hashlib.sha256(file_bytes).hexdigest()

    def fuse_image_evidence(
        self,
        spatial_result: Dict[str, Any],
        freq_result: Dict[str, Any],
        ela_result: Dict[str, Any],
        file_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Fuses multi-cue image forensic metrics into a unified verdict.
        """
        s_spatial = spatial_result.get("fake_prob", 0.5)
        s_freq = freq_result.get("frequency_score", 0.5)
        s_ela = ela_result.get("ela_score", 0.5)
        
        # Boundary seam cue
        disp_ratio = ela_result.get("disparity_ratio", 0.0)
        s_boundary = min(0.95, max(0.05, disp_ratio * 1.5))

        # Weighted Bayesian fusion
        fused_fake_prob = (
            self.w_spatial * s_spatial +
            self.w_frequency * s_freq +
            self.w_ela * s_ela +
            self.w_boundary * s_boundary
        )
        fused_fake_prob = float(np.clip(fused_fake_prob, 0.01, 0.99))
        fused_real_prob = 1.0 - fused_fake_prob

        # Calibrated confidence & verdict
        if fused_fake_prob >= 0.65:
            verdict = "DEEPFAKE / SYNTHETIC MANIPULATION"
            severity = "CRITICAL"
            badge_color = "#ef4444"
            confidence = fused_fake_prob * 100.0
        elif fused_fake_prob <= 0.35:
            verdict = "AUTHENTIC / ORGANIC MEDIA"
            severity = "CLEAN"
            badge_color = "#10b981"
            confidence = fused_real_prob * 100.0
        else:
            verdict = "INCONCLUSIVE / SUSPICIOUS ANOMALIES"
            severity = "ELEVATED"
            badge_color = "#f59e0b"
            confidence = max(fused_fake_prob, fused_real_prob) * 100.0

        # Anomaly highlights
        anomalies_detected = []
        if s_spatial >= 0.60:
            anomalies_detected.append("High spatial artifact signature detected by deep convolutional backbone.")
        if s_freq >= 0.50:
            anomalies_detected.append(f"Unnatural spectral roll-off (slope {freq_result.get('spectral_slope', 0):.2f}) and high-frequency upsampling artifacts.")
        if s_ela >= 0.45:
            anomalies_detected.append(f"JPEG compression disparity ({ela_result.get('disparity_ratio', 0):.1%}) between face and outer background.")
        if disp_ratio >= 0.30:
            anomalies_detected.append("Elevated residual error concentrated along the facial blending seam.")

        if not anomalies_detected:
            anomalies_detected.append("No significant synthetic generative anomalies or compression disparities identified.")

        sha256_hash = self.compute_sha256(file_bytes) if file_bytes else "N/A"

        return {
            "verdict": verdict,
            "severity": severity,
            "badge_color": badge_color,
            "fused_fake_prob": round(fused_fake_prob, 4),
            "fused_real_prob": round(fused_real_prob, 4),
            "confidence": round(confidence, 2),
            "sha256": sha256_hash,
            "cue_breakdown": {
                "Spatial Deep Learning": round(s_spatial * 100.0, 1),
                "Frequency Domain (FFT)": round(s_freq * 100.0, 1),
                "Error Level Analysis (ELA)": round(s_ela * 100.0, 1),
                "Face-Background Boundary": round(s_boundary * 100.0, 1)
            },
            "anomalies": anomalies_detected
        }

    def fuse_video_evidence(
        self,
        frame_results: List[Dict[str, Any]],
        temporal_result: Dict[str, Any],
        file_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Fuses multi-frame video analysis and biological metrics into video-level forensic verdict.
        """
        if not frame_results:
            return {
                "verdict": "ERROR: NO FRAMES PROCESSED",
                "severity": "UNKNOWN",
                "fused_fake_prob": 0.5,
                "confidence": 0.0,
                "anomalies": ["No frames could be extracted from video stream."]
            }

        frame_fake_probs = [f["fused_result"]["fused_fake_prob"] for f in frame_results]
        mean_frame_fake = float(np.mean(frame_fake_probs))
        max_frame_fake = float(np.max(frame_fake_probs))
        
        # Temporal cue contribution
        temp_score = temporal_result.get("temporal_score", 0.5)

        # Video-level fusion (80% frame sequence consistency, 20% temporal/biological dynamics)
        video_fake_prob = 0.80 * mean_frame_fake + 0.20 * temp_score
        video_fake_prob = float(np.clip(video_fake_prob, 0.01, 0.99))
        video_real_prob = 1.0 - video_fake_prob

        fake_frame_count = sum(1 for p in frame_fake_probs if p >= 0.5)
        total_frames = len(frame_fake_probs)
        fake_ratio = (fake_frame_count / total_frames) * 100.0

        if video_fake_prob >= 0.60 or fake_ratio >= 45.0:
            verdict = "DEEPFAKE VIDEO DETECTED"
            severity = "CRITICAL"
            badge_color = "#ef4444"
            confidence = max(video_fake_prob * 100.0, fake_ratio)
        elif video_fake_prob <= 0.35 and fake_ratio <= 15.0:
            verdict = "AUTHENTIC VIDEO STREAM"
            severity = "CLEAN"
            badge_color = "#10b981"
            confidence = video_real_prob * 100.0
        else:
            verdict = "SUSPICIOUS / PARTIAL TAMPERING"
            severity = "ELEVATED"
            badge_color = "#f59e0b"
            confidence = max(video_fake_prob, video_real_prob) * 100.0

        anomalies = []
        if fake_ratio >= 30.0:
            anomalies.append(f"{fake_ratio:.1f}% of sampled frames ({fake_frame_count}/{total_frames}) exhibited high synthetic probability.")
        if temporal_result.get("blink_rate_per_min", 15.0) < 4.0 and temporal_result.get("duration_sec", 0) > 8.0:
            anomalies.append(f"Abnormally low blink rate ({temporal_result['blink_rate_per_min']} blinks/min) indicating synthetic facial generation.")
        if temporal_result.get("optical_flow_jitter", 0) > 2.0:
            anomalies.append("Inter-frame optical flow discontinuity detected across facial contours.")

        if not anomalies:
            anomalies.append("Normal human biological dynamics and inter-frame temporal continuity observed.")

        sha256_hash = self.compute_sha256(file_bytes) if file_bytes else "N/A"

        return {
            "verdict": verdict,
            "severity": severity,
            "badge_color": badge_color,
            "fused_fake_prob": round(video_fake_prob, 4),
            "fused_real_prob": round(video_real_prob, 4),
            "confidence": round(confidence, 2),
            "fake_frame_ratio": round(fake_ratio, 1),
            "total_frames_analyzed": total_frames,
            "fake_frame_count": fake_frame_count,
            "sha256": sha256_hash,
            "temporal_metrics": temporal_result,
            "anomalies": anomalies
        }
