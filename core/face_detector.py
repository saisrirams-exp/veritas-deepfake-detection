"""
Face Detection, Alignment, and Facial Region Extraction Module
Provides accurate face localization, facial boundary expansion, and landmark extraction.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image

class FaceForensicExtractor:
    """
    Locates faces, extracts face crops with blending margins,
    and isolates high-risk facial sub-regions (eyes, mouth, boundary).
    """

    def __init__(self):
        # Load OpenCV Haar cascade classifiers as fast, robust base detector
        cv_data_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cv_data_dir + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv_data_dir + "haarcascade_eye.xml")
        self.mouth_cascade = cv2.CascadeClassifier(cv_data_dir + "haarcascade_smile.xml")

    def detect_faces(self, image: np.ndarray, margin: float = 0.2) -> List[Dict[str, Any]]:
        """
        Detect all faces in an image and return bounding boxes with margins and sub-crops.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            margin: Extra margin around face box to capture blending seams (default 20%)
            
        Returns:
            List of dictionaries containing face bounding boxes, crops, and landmarks
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect raw face boxes
        detected = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []
        for idx, (x, y, fw, fh) in enumerate(detected):
            # Expand box to include hair, neck, and blending borders
            mx = int(fw * margin)
            my = int(fh * margin)
            
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)

            face_crop = image[y1:y2, x1:x2]
            face_gray = gray[y1:y2, x1:x2]

            # Detect eyes within the face crop
            eyes = self.eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(15, 15)
            )

            # Sort eyes horizontally
            eyes_list = []
            for (ex, ey, ew, eh) in eyes:
                # Filter out false positives in lower half of face
                if ey < (y2 - y1) * 0.65:
                    eyes_list.append({
                        "rel_box": (ex, ey, ew, eh),
                        "abs_box": (x1 + ex, y1 + ey, ew, eh),
                        "center": (x1 + ex + ew // 2, y1 + ey + eh // 2)
                    })

            # Create boundary mask highlighting transition between face and perimeter
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.ellipse(
                mask,
                ((x2 - x1) // 2, (y2 - y1) // 2),
                ((x2 - x1) // 2 - 4, (y2 - y1) // 2 - 4),
                0, 0, 360, 255, 3
            )

            results.append({
                "face_id": idx,
                "box": (x1, y1, x2 - x1, y2 - y1),
                "box_tight": (x, y, fw, fh),
                "face_crop": face_crop,
                "eyes": eyes_list,
                "boundary_mask": mask,
                "confidence": 0.95
            })

        # If no face is detected, provide full-frame center region as fallback
        if not results:
            # Fallback center crop
            cx, cy = w // 2, h // 2
            size = min(w, h)
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, x1 + size)
            y2 = min(h, y1 + size)
            crop = image[y1:y2, x1:x2]

            results.append({
                "face_id": 0,
                "box": (x1, y1, x2 - x1, y2 - y1),
                "box_tight": (x1, y1, x2 - x1, y2 - y1),
                "face_crop": crop,
                "eyes": [],
                "boundary_mask": np.zeros(crop.shape[:2], dtype=np.uint8),
                "confidence": 0.40,
                "is_fallback": True
            })

        return results

    def draw_forensic_hud(self, image: np.ndarray, faces: List[Dict[str, Any]], fake_prob: Optional[float] = None) -> np.ndarray:
        """
        Draws high-tech, cyberpunk forensic HUD overlays onto the image:
        Targeting brackets, biometric markers, eye tracking lines, and verdict tags.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        color = (0, 240, 255)  # Cyan default
        if fake_prob is not None:
            if fake_prob >= 0.5:
                # Crimson / Red for fake
                color = (255, 50, 70)
            else:
                # Vibrant Emerald green for authentic
                color = (40, 230, 120)

        for face in faces:
            x, y, fw, fh = face["box"]
            corner_len = min(24, fw // 5, fh // 5)

            # Draw tactical corner brackets
            t = 2
            # Top-left
            cv2.line(annotated, (x, y), (x + corner_len, y), color, t)
            cv2.line(annotated, (x, y), (x, y + corner_len), color, t)
            # Top-right
            cv2.line(annotated, (x + fw, y), (x + fw - corner_len, y), color, t)
            cv2.line(annotated, (x + fw, y), (x + fw, y + corner_len), color, t)
            # Bottom-left
            cv2.line(annotated, (x, y + fh), (x + corner_len, y + fh), color, t)
            cv2.line(annotated, (x, y + fh), (x, y + fh - corner_len), color, t)
            # Bottom-right
            cv2.line(annotated, (x + fw, y + fh), (x + fw - corner_len, y + fh), color, t)
            cv2.line(annotated, (x + fw, y + fh), (x + fw, y + fh - corner_len), color, t)

            # Draw subtle dashed bounding rectangle
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x, y), (x + fw, y + fh), color, 1)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

            # Draw eye detection markers
            for eye in face.get("eyes", []):
                ex, ey = eye["center"]
                cv2.circle(annotated, (ex, ey), 4, (0, 255, 255), -1)
                cv2.circle(annotated, (ex, ey), 9, (0, 255, 255), 1)

            # Draw biometric ID tag
            tag_text = f"TARGET #0{face['face_id'] + 1} | SCAN: ACTIVE"
            if face.get("is_fallback"):
                tag_text = "GLOBAL FRAME ANALYSIS (NO FACE LOCK)"
            elif fake_prob is not None:
                tag_text = f"FACE_ID_0{face['face_id'] + 1} | SYNTHETIC: {fake_prob*100:.1f}%"

            cv2.rectangle(annotated, (x, max(0, y - 22)), (x + len(tag_text) * 9 + 8, y), (15, 20, 30), -1)
            cv2.putText(annotated, tag_text, (x + 4, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        return annotated
