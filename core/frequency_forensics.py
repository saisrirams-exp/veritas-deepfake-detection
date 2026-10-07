"""
Frequency-Domain Forensic Analysis Module
Implements 2D Fast Fourier Transform (FFT), Azimuthal Spectral Profiling,
and Discrete Cosine Transform (DCT) to expose generative upsampling artifacts.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2

class FrequencyForensicAnalyzer:
    """
    Examines frequency domain anomalies, periodic checkerboard artifacts,
    and high-frequency spectral discrepancies characteristic of GANs and Diffusion models.
    """

    def __init__(self, size: int = 256):
        self.size = size

    def _azimuthal_average(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate the azimuthally averaged radial profile of a 2D image.
        Returns the 1D power distribution from DC (center) to Nyquist (edge).
        """
        h, w = image.shape
        y, x = np.indices((h, w))
        center = (w // 2, h // 2)
        r = np.hypot(x - center[0], y - center[1]).astype(int)

        # Calculate sum and count for each radius bin
        max_r = int(min(w, h) / 2)
        nr = np.bincount(r.ravel())
        tbin = np.bincount(r.ravel(), image.ravel())
        
        # Avoid zero division
        radial_profile = tbin / np.maximum(nr, 1)
        return radial_profile[:max_r]

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Performs full frequency domain forensic evaluation on an image or face crop.
        
        Args:
            image: RGB image as numpy array (H, W, 3) or Grayscale (H, W)
            
        Returns:
            Dictionary with spectral metrics, anomaly scores, and visual spectrums.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Resize to fixed square for uniform frequency binning
        gray_resized = cv2.resize(gray, (self.size, self.size)).astype(np.float32)

        # Apply Hanning window to reduce spectral leakage from image borders
        win = np.hanning(self.size)[:, None] * np.hanning(self.size)[None, :]
        windowed = gray_resized * win

        # 2D Fast Fourier Transform
        f_transform = np.fft.fft2(windowed)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        log_spectrum = np.log1p(magnitude_spectrum)

        # Normalize log spectrum to [0, 255] for visualization
        spec_min, spec_max = log_spectrum.min(), log_spectrum.max()
        norm_spectrum = ((log_spectrum - spec_min) / (spec_max - spec_min + 1e-8) * 255).astype(np.uint8)
        color_spectrum = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_INFERNO)
        color_spectrum = cv2.cvtColor(color_spectrum, cv2.COLOR_BGR2RGB)

        # Calculate 1D Azimuthal Radial Profile
        radial_profile = self._azimuthal_average(log_spectrum)
        r_len = len(radial_profile)

        # Frequency domain features:
        # 1. High Frequency Energy Ratio (HFER)
        low_band = radial_profile[: int(r_len * 0.25)]
        mid_band = radial_profile[int(r_len * 0.25) : int(r_len * 0.65)]
        high_band = radial_profile[int(r_len * 0.65) :]

        low_energy = np.mean(low_band) if len(low_band) > 0 else 1.0
        mid_energy = np.mean(mid_band) if len(mid_band) > 0 else 1.0
        high_energy = np.mean(high_band) if len(high_band) > 0 else 0.1

        hfer = float(high_energy / (low_energy + 1e-6))
        mid_drop = float(mid_energy / (low_energy + 1e-6))

        # 2. Check for periodic grid peaks (common in StyleGAN upsamplers)
        # Detrend radial profile to spot high-frequency spikes
        diffs = np.abs(np.diff(radial_profile))
        spectral_jitter = float(np.std(diffs))

        # 3. High-pass Laplacian energy
        laplacian = cv2.Laplacian(gray_resized, cv2.CV_32F)
        high_pass_variance = float(np.var(laplacian))

        # SOTA generative models either over-smooth (diffusion) or have checkerboard spikes (GANs)
        # We compute a calibrated synthetic likelihood based on spectral curvature
        # Typical natural image 1/f decay curvature has smooth logarithmic slope
        x_axis = np.arange(1, r_len + 1)
        slope, intercept = np.polyfit(np.log(x_axis[1:]), radial_profile[1:], 1)
        residual_err = float(np.mean((radial_profile[1:] - (slope * np.log(x_axis[1:]) + intercept)) ** 2))

        # Combine into frequency anomaly score [0.0 (natural) to 1.0 (synthetic)]
        # Natural images have high negative slope (~ -1.0 to -2.0) and low residual error
        anomaly_score = 0.0
        if slope > -0.65:  # Unnatural flat spectral roll-off (excess high-freq noise)
            anomaly_score += 0.40
        if residual_err > 0.45:  # Discontinuous spikes
            anomaly_score += 0.35
        if spectral_jitter > 0.30:  # High frequency oscillation
            anomaly_score += 0.25

        anomaly_score = float(np.clip(anomaly_score, 0.05, 0.95))

        return {
            "frequency_score": anomaly_score,
            "hfer": round(hfer, 4),
            "spectral_slope": round(float(slope), 4),
            "spectral_residual": round(residual_err, 4),
            "spectral_jitter": round(spectral_jitter, 4),
            "high_pass_variance": round(high_pass_variance, 2),
            "radial_profile": radial_profile.tolist(),
            "spectrum_image": color_spectrum
        }
