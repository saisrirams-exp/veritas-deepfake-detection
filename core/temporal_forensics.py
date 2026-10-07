"""
Temporal & Biological Video Forensic Analysis Module
Evaluates physiological signals: Eye-Blink Dynamics, Inter-Frame Landmark Jitter,
and Optical Flow Temporal Coherence across video streams.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

class TemporalForensicAnalyzer:
    """
    Examines biological signals and temporal consistency across consecutive video frames.
    """

    def __init__(self):
        self.prev_gray = None
        self.prev_landmarks = None

    def analyze_frame_sequence(self, frames_data: List[Dict[str, Any]], fps: float = 30.0) -> Dict[str, Any]:
        """
        Analyzes a sequence of processed video frames for temporal anomalies.
        
        Args:
            frames_data: List of frame dicts with keys:
                         'frame_idx', 'face_crop', 'eyes', 'spatial_fake_prob', 'confidence'
            fps: Frame rate of the input video
            
        Returns:
            Dictionary with temporal consistency metrics, blink counts, and jitter scores.
        """
        if not frames_data:
            return {
                "temporal_score": 0.5,
                "blink_count": 0,
                "blink_rate_per_min": 0.0,
                "jitter_variance": 0.0,
                "optical_flow_discontinuity": 0.0,
                "verdict": "INSUFFICIENT_FRAMES"
            }

        total_frames = len(frames_data)
        duration_sec = max(1.0, total_frames / fps)

        # 1. Spatial fake probability trajectory
        fake_probs = [f.get("spatial_fake_prob", 0.5) for f in frames_data]
        prob_std = float(np.std(fake_probs))
        max_prob = float(np.max(fake_probs))
        mean_prob = float(np.mean(fake_probs))

        # 2. Inter-frame optical flow discontinuity
        flow_discontinuities = []
        for i in range(1, total_frames):
            prev_crop = frames_data[i - 1].get("face_crop")
            curr_crop = frames_data[i].get("face_crop")

            if prev_crop is not None and curr_crop is not None:
                try:
                    # Resize to uniform thumbnail
                    p_g = cv2.cvtColor(cv2.resize(prev_crop, (128, 128)), cv2.COLOR_RGB2GRAY)
                    c_g = cv2.cvtColor(cv2.resize(curr_crop, (128, 128)), cv2.COLOR_RGB2GRAY)
                    
                    # Compute Farneback optical flow
                    flow = cv2.calcOpticalFlowFarneback(p_g, c_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_discontinuities.append(float(np.mean(mag)))
                except Exception:
                    continue

        mean_flow_disp = float(np.mean(flow_discontinuities)) if flow_discontinuities else 0.0
        flow_jitter = float(np.std(flow_discontinuities)) if flow_discontinuities else 0.0

        # 3. Eye-Blink Dynamic Estimation
        # In deepfakes, eye detection often drops out or blinks are unnaturally rapid/absent
        eye_counts = [len(f.get("eyes", [])) for f in frames_data]
        blinks_detected = 0
        in_blink = False
        blink_lengths = []
        current_len = 0

        for count in eye_counts:
            # If 0 eyes detected in an otherwise detected face, possible blink
            if count == 0:
                if not in_blink:
                    in_blink = True
                    current_len = 1
                else:
                    current_len += 1
            else:
                if in_blink:
                    in_blink = False
                    # Normal blink lasts between 2 and 12 frames at 30fps
                    if 2 <= current_len <= 15:
                        blinks_detected += 1
                        blink_lengths.append(current_len)
                    current_len = 0

        blinks_per_min = (blinks_detected / duration_sec) * 60.0

        # Biological Anomaly Evaluation
        # Normal human blink rate: 12-24 blinks/min in conversation
        biological_fake_cues = 0.0
        if duration_sec > 8.0:
            if blinks_per_min < 4.0:  # Unnatural lack of blinking (mannequin effect)
                biological_fake_cues += 0.35
            elif blinks_per_min > 45.0:  # Excessive sporadic flickering
                biological_fake_cues += 0.30

        # Temporal jitter cue
        if flow_jitter > 2.5:  # Inter-frame facial warping / flickering
            biological_fake_cues += 0.35

        # Rolling probability volatility
        if prob_std > 0.25:
            biological_fake_cues += 0.30

        temporal_score = float(np.clip(biological_fake_cues, 0.05, 0.95))

        return {
            "temporal_score": round(temporal_score, 4),
            "mean_fake_prob": round(mean_prob, 4),
            "max_fake_prob": round(max_prob, 4),
            "prob_volatility": round(prob_std, 4),
            "blinks_detected": blinks_detected,
            "blink_rate_per_min": round(blinks_per_min, 1),
            "optical_flow_jitter": round(flow_jitter, 3),
            "duration_sec": round(duration_sec, 2),
            "total_frames_analyzed": total_frames,
            "flow_discontinuity_curve": [round(x, 2) for x in flow_discontinuities]
        }
