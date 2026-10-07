"""
VERITAS Enterprise Python SDK
Official client library for integrating VERITAS Deepfake Cyber-Forensics
into enterprise workflows, SOC automation, KYC verification, and e-Discovery pipelines.
"""

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class VeritasAnalysisResult:
    """Encapsulates image or video forensic evaluation results."""
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.media_name: str = data.get("media_name", "")
        self.verdict: str = data.get("verdict", "UNKNOWN")
        self.severity: str = data.get("severity", "LOW")
        self.fake_probability: float = float(data.get("fused_fake_prob", 0.0) or data.get("fake_frame_ratio", 0.0))
        self.confidence: float = float(data.get("confidence", 0.0))
        self.sha256: str = data.get("sha256", "")
        self.anomalies: List[str] = data.get("anomalies", [])
        
        attr = data.get("generator_attribution", {})
        self.primary_generator: str = attr.get("primary_attribution", "N/A")
        self.generator_confidence: float = attr.get("attribution_confidence", 0.0)

    def is_synthetic(self) -> bool:
        return self.fake_probability >= 0.50

    def __repr__(self) -> str:
        return f"<VeritasResult {self.media_name}: {self.verdict} ({self.fake_probability*100:.1f}%)>"


class VeritasAudioResult:
    """Encapsulates audio voice clone analysis results."""
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.media_name: str = data.get("media_name", "")
        self.verdict: str = data.get("verdict", "UNKNOWN")
        self.synthetic_probability: float = float(data.get("synthetic_probability", 0.0))
        self.confidence: float = float(data.get("confidence", 0.0))
        self.sha256: str = data.get("sha256", "")
        self.anomalies: List[str] = data.get("anomalies", [])
        self.metrics: Dict[str, Any] = data.get("metrics", {})

    def is_clone(self) -> bool:
        return self.synthetic_probability >= 0.50

    def __repr__(self) -> str:
        return f"<VeritasAudioResult {self.media_name}: {self.verdict} ({self.synthetic_probability*100:.1f}%)>"


class VeritasCase:
    """Encapsulates an investigative case."""
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.case_id: str = data.get("case_id", "")
        self.title: str = data.get("title", "")
        self.investigator: str = data.get("investigator", "")
        self.organization: str = data.get("organization", "")
        self.created_utc: str = data.get("created_utc", "")
        self.exhibits: List[Dict[str, Any]] = data.get("exhibits", [])

    def __repr__(self) -> str:
        return f"<VeritasCase {self.case_id}: {self.title} ({len(self.exhibits)} exhibits)>"


class VeritasClient:
    """
    Enterprise client for the VERITAS SOTA Deepfake Forensics platform.
    
    Example usage:
        client = VeritasClient("http://localhost:8000")
        
        # Analyze portrait photo
        result = client.analyze_image("suspect_ceo.jpg")
        print(result.verdict, result.primary_generator)
        
        # Create case & attach exhibit
        case = client.create_case("CEO Impersonation", "Analyst Jane", "FinSec SOC")
        client.attach_evidence(case.case_id, result)
        client.download_case_pdf(case.case_id, "Master_Dossier.pdf")
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "Veritas-Python-SDK/2.1.0"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def get_telemetry(self) -> Dict[str, Any]:
        """Fetch system hardware accelerator, backbone specs, and status."""
        req = urllib.request.Request(f"{self.base_url}/api/telemetry", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def analyze_image(self, file_path_or_bytes: Union[str, Path, bytes], filename: str = "evidence.jpg") -> VeritasAnalysisResult:
        """
        Submits an image for multi-modal spatial-frequency cross-attention forensic evaluation.
        """
        if isinstance(file_path_or_bytes, (str, Path)):
            p = Path(file_path_or_bytes)
            raw_bytes = p.read_bytes()
            fname = p.name
        else:
            raw_bytes = file_path_or_bytes
            fname = filename

        boundary = "----VeritasSDKFormBoundary" + str(int(urllib.request.time.time()))
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(raw_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        headers = self._headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        req = urllib.request.Request(f"{self.base_url}/api/analyze/image", data=bytes(body), headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return VeritasAnalysisResult(data)

    def analyze_audio(self, file_path_or_bytes: Union[str, Path, bytes], filename: str = "voice.wav") -> VeritasAudioResult:
        """
        Submits an audio file for neural vocoder and voice clone detection.
        """
        if isinstance(file_path_or_bytes, (str, Path)):
            p = Path(file_path_or_bytes)
            raw_bytes = p.read_bytes()
            fname = p.name
        else:
            raw_bytes = file_path_or_bytes
            fname = filename

        boundary = "----VeritasAudioBoundary" + str(int(urllib.request.time.time()))
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: audio/wav\r\n\r\n")
        body.extend(raw_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        headers = self._headers()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        req = urllib.request.Request(f"{self.base_url}/api/analyze/audio", data=bytes(body), headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return VeritasAudioResult(data)

    def create_case(self, title: str, investigator: str, organization: str) -> VeritasCase:
        """Opens a new digital forensics case."""
        payload = json.dumps({"title": title, "investigator": investigator, "organization": organization}).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{self.base_url}/api/cases", data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            return VeritasCase(json.loads(resp.read().decode("utf-8")))

    def list_cases(self) -> List[VeritasCase]:
        """Lists all existing cases."""
        req = urllib.request.Request(f"{self.base_url}/api/cases", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [VeritasCase(c) for c in data.get("cases", [])]

    def attach_evidence(self, case_id: str, analysis: Union[VeritasAnalysisResult, VeritasAudioResult, Dict[str, Any]], notes: str = "") -> Dict[str, Any]:
        """Attaches an analyzed piece of evidence to an active case."""
        if isinstance(analysis, (VeritasAnalysisResult, VeritasAudioResult)):
            raw = analysis.raw
        else:
            raw = analysis

        threat = "Unknown"
        if "generator_attribution" in raw:
            threat = raw["generator_attribution"].get("primary_attribution", "Unknown")

        payload = json.dumps({
            "media_name": raw.get("media_name", "exhibit"),
            "verdict": raw.get("verdict", "UNKNOWN"),
            "confidence": float(raw.get("confidence", 0.0)),
            "sha256": raw.get("sha256", "UNKNOWN"),
            "threat_actor": threat,
            "notes": notes
        }).encode("utf-8")

        headers = self._headers()
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(f"{self.base_url}/api/cases/{case_id}/exhibits", data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def download_case_pdf(self, case_id: str, output_path: Union[str, Path]) -> Path:
        """Downloads the certified multi-page Judicial Master Evidence Dossier PDF."""
        req = urllib.request.Request(f"{self.base_url}/api/cases/{case_id}/export/pdf", headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            out = Path(output_path)
            out.write_bytes(content)
            return out
