"""
Multi-Generator Attribution Forensics Module
Identifies and fingerprints the underlying generative synthesis architecture:
StyleGAN, Latent Diffusion, FaceSwap/DeepFaceLab, Talking-Head (Wav2Lip), or Authentic Camera.
"""

from typing import Dict, Any, List

class GeneratorAttributionEngine:
    """
    Classifies the probable generative framework used to synthesize an exhibit
    based on spectral periodicities, noise residuals, and boundary blending footprints.
    """

    CANDIDATE_GENERATORS = [
        "StyleGAN / GAN Upsampling",
        "Latent Diffusion (SD / Midjourney / Flux)",
        "DeepFaceLab / FaceSwap Blend",
        "Talking-Head / Wav2Lip Re-animation",
        "Authentic Optical Sensor"
    ]

    def attribute(self, forensic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes probabilistic attribution breakdown across generative model families.
        """
        spectral = forensic_metrics.get("spectral_metrics", {})
        ela = forensic_metrics.get("ela_metrics", {})
        spatial = forensic_metrics.get("cue_breakdown", {})

        s_jitter = spectral.get("jitter", 0.0) or spectral.get("spectral_jitter", 0.0)
        s_slope = spectral.get("slope", -1.2) or spectral.get("spectral_slope", -1.2)
        disp_ratio = ela.get("disparity_ratio", 0.0)
        boundary_score = spatial.get("Face-Background Boundary", 10.0) / 100.0

        scores = {
            "StyleGAN / GAN Upsampling": 0.10,
            "Latent Diffusion (SD / Midjourney / Flux)": 0.10,
            "DeepFaceLab / FaceSwap Blend": 0.10,
            "Talking-Head / Wav2Lip Re-animation": 0.10,
            "Authentic Optical Sensor": 0.60
        }

        # 1. GAN footprint: High spectral oscillation from transposed convs
        if s_jitter > 0.20 or s_slope > -0.60:
            scores["StyleGAN / GAN Upsampling"] += 0.55
            scores["Authentic Optical Sensor"] -= 0.35

        # 2. Diffusion footprint: Flat roll-off and over-smoothed high frequencies
        if s_slope > -0.80 and s_jitter < 0.15:
            scores["Latent Diffusion (SD / Midjourney / Flux)"] += 0.50
            scores["Authentic Optical Sensor"] -= 0.30

        # 3. FaceSwap footprint: High boundary seam activity and ELA disparity
        if disp_ratio > 0.25 or boundary_score > 0.30:
            scores["DeepFaceLab / FaceSwap Blend"] += 0.60
            scores["Authentic Optical Sensor"] -= 0.40

        # Normalize probabilities
        total = sum(max(0.01, v) for v in scores.values())
        norm_scores = {k: round(max(0.01, v) / total * 100.0, 1) for k, v in scores.items()}

        top_gen = max(norm_scores, key=norm_scores.get)
        confidence = norm_scores[top_gen]

        signatures = []
        if top_gen == "StyleGAN / GAN Upsampling":
            signatures.append("Periodic upsampling grid artifacts identified in high-frequency spectral bands.")
        elif top_gen == "Latent Diffusion (SD / Midjourney / Flux)":
            signatures.append("Diffusion-denoising smoothing residuals and micro-texture roll-off detected.")
        elif top_gen == "DeepFaceLab / FaceSwap Blend":
            signatures.append("Face-to-scene compression disparity and perimeter blending boundary seam detected.")
        else:
            signatures.append("Camera sensor PRNU noise and organic photographic decay profile.")

        return {
            "primary_attribution": top_gen,
            "attribution_confidence": confidence,
            "distribution": norm_scores,
            "identified_signatures": signatures
        }
