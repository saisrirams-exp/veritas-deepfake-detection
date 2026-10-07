"""
Enterprise SIEM & SOC Alert Dispatcher
Dispatches real-time forensic incident alerts to enterprise SIEM/SOC platforms
(Splunk, Datadog, Elastic, Slack, MS Teams, PagerDuty, or custom webhooks)
with cryptographic HMAC-SHA256 non-repudiation signatures.
"""

import json
import time
import hmac
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List


class SIEMAlertDispatcher:
    """
    Manages SIEM webhook endpoints and dispatches threat incident notifications.
    """

    def __init__(self, config_path: str = "data/siem_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_config()

    def load_config(self):
        default_config = {
            "enabled": False,
            "webhook_url": "",
            "alert_threshold": 0.70,  # Alert when fake_prob >= 70%
            "hmac_secret": "veritas_enterprise_sec_key_2026",
            "channels": {
                "slack_compatible": True,
                "splunk_cef": False,
                "generic_json": True
            },
            "recent_dispatches": []
        }
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    def update_config(self, enabled: bool, webhook_url: str, alert_threshold: float, hmac_secret: str) -> Dict[str, Any]:
        self.config["enabled"] = enabled
        self.config["webhook_url"] = webhook_url.strip()
        self.config["alert_threshold"] = float(alert_threshold)
        if hmac_secret:
            self.config["hmac_secret"] = hmac_secret.strip()
        self.save_config()
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        masked_secret = ""
        if self.config.get("hmac_secret"):
            s = self.config["hmac_secret"]
            masked_secret = s[:3] + "*" * (len(s) - 5) + s[-2:] if len(s) > 5 else "***"
        return {
            "enabled": self.config.get("enabled", False),
            "webhook_url": self.config.get("webhook_url", ""),
            "alert_threshold": self.config.get("alert_threshold", 0.70),
            "hmac_secret_masked": masked_secret,
            "dispatches_count": len(self.config.get("recent_dispatches", [])),
            "recent_dispatches": self.config.get("recent_dispatches", [])[-5:]
        }

    def compute_signature(self, payload_bytes: bytes) -> str:
        secret = self.config.get("hmac_secret", "veritas_enterprise_sec_key_2026").encode("utf-8")
        return "sha256=" + hmac.new(secret, payload_bytes, hashlib.sha256).hexdigest()

    def dispatch_incident_alert(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches an incident payload to configured SIEM webhook if threshold is breached.
        """
        fake_prob = float(incident_data.get("fused_fake_prob", 0.0) or incident_data.get("synthetic_probability", 0.0))
        threshold = float(self.config.get("alert_threshold", 0.70))

        if not self.config.get("enabled", False) or not self.config.get("webhook_url"):
            return {"dispatched": False, "reason": "Webhook disabled or URL unset"}

        if fake_prob < threshold:
            return {"dispatched": False, "reason": f"Probability {fake_prob:.2f} below alert threshold {threshold:.2f}"}

        payload = {
            "event_type": "SECURITY_INCIDENT_DEEPFAKE_DETECTED",
            "severity": "CRITICAL" if fake_prob > 0.85 else "HIGH",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threat_actor": incident_data.get("generator_attribution", {}).get("primary_attribution", "Unknown AI Synthesizer"),
            "evidence": {
                "media_name": incident_data.get("media_name", "evidence_item"),
                "sha256": incident_data.get("sha256", "UNKNOWN"),
                "fake_probability": round(fake_prob * 100.0, 1),
                "verdict": incident_data.get("verdict", "SYNTHETIC"),
                "confidence": incident_data.get("confidence", 0.0),
                "anomalies": incident_data.get("anomalies", [])[:3]
            },
            "investigation_platform": "VERITAS SOTA Cyber-Forensics v2.0"
        }

        # Format for Slack if webhook is slack format
        webhook_url = self.config["webhook_url"]
        if "slack.com" in webhook_url:
            slack_msg = {
                "text": f":warning: *CRITICAL DEEPFAKE DETECTED* - {payload['evidence']['media_name']}\n"
                        f"*Verdict:* `{payload['evidence']['verdict']}` ({payload['evidence']['fake_probability']}% probability)\n"
                        f"*Threat Generator:* `{payload['threat_actor']}`\n"
                        f"*SHA-256:* `{payload['evidence']['sha256']}`"
            }
            body = json.dumps(slack_msg).encode("utf-8")
        else:
            body = json.dumps(payload, indent=2).encode("utf-8")

        sig = self.compute_signature(body)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VERITAS-SIEM-Agent/2.0",
            "X-VERITAS-Signature": sig
        }

        status_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "media_name": payload["evidence"]["media_name"],
            "fake_prob": round(fake_prob * 100.0, 1),
            "verdict": payload["evidence"]["verdict"]
        }

        try:
            req = urllib.request.Request(webhook_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                status_record["status"] = "DELIVERED"
                status_record["http_status"] = resp.status
        except Exception as e:
            status_record["status"] = "FAILED"
            status_record["error"] = str(e)

        dispatches = self.config.get("recent_dispatches", [])
        dispatches.append(status_record)
        self.config["recent_dispatches"] = dispatches[-20:]
        self.save_config()

        return {"dispatched": True, "delivery_record": status_record}

    def test_ping(self) -> Dict[str, Any]:
        """Send a test canary alert payload to verify webhook reachability."""
        canary = {
            "media_name": "canary_verification_test.jpg",
            "fused_fake_prob": 0.99,
            "verdict": "CANARY TEST INCIDENT",
            "confidence": 99.9,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "generator_attribution": {"primary_attribution": "Canary Automated Ping"},
            "anomalies": ["This is an automated test ping from the VERITAS SIEM dispatcher."]
        }
        orig_thresh = self.config.get("alert_threshold", 0.70)
        self.config["alert_threshold"] = 0.0
        try:
            res = self.dispatch_incident_alert(canary)
        finally:
            self.config["alert_threshold"] = orig_thresh
            self.save_config()
        return res
