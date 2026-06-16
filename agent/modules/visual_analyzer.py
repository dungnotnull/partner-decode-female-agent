"""
visual_analyzer.py — Facial action unit detection and expression classification.

Uses MediaPipe FaceMesh 468 landmarks to compute facial action units (AU),
detect microexpressions (<200ms AU transitions), and classify facial expressions.
Degrades gracefully when MediaPipe is unavailable.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe landmark indices for key facial regions
# These correspond to the 468-point FaceMesh model
LANDMARK_INDICES = {
    # Brow landmarks
    "left_inner_brow": 336,
    "left_outer_brow": 334,
    "right_inner_brow": 107,
    "right_outer_brow": 105,
    # Eye landmarks
    "left_eye_top": 386,
    "left_eye_bottom": 374,
    "left_eye_left": 362,
    "left_eye_right": 263,
    "right_eye_top": 159,
    "right_eye_bottom": 145,
    "right_eye_left": 133,
    "right_eye_right": 33,
    # Nose bridge
    "nose_bridge": 6,
    # Mouth landmarks
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip_top": 13,
    "lower_lip_bottom": 14,
    "lip_left_corner": 291,
    "lip_right_corner": 61,
    # Chin
    "chin": 18,
    # Cheek
    "left_cheek": 425,
    "right_cheek": 205,
}

# Facial Action Coding System (FACS) AU descriptions
AU_DESCRIPTIONS = {
    "AU1": "Inner Brow Raise (surprise, concern, sadness)",
    "AU2": "Outer Brow Raise (surprise, fear)",
    "AU4": "Brow Lowerer (anger, confusion, concentration)",
    "AU6": "Cheek Raiser (genuine smile — Duchenne marker)",
    "AU12": "Lip Corner Puller (smile)",
    "AU15": "Lip Corner Depressor (sadness)",
    "AU17": "Chin Raiser (sadness, uncertainty)",
    "AU20": "Lip Stretcher (fear)",
    "AU26": "Jaw Drop (surprise, open-mouthed speech)",
    "AU45": "Blink",
}

# Expression classification mapping (AU combinations → expression)
EXPRESSION_AU_MAP = {
    "happy": {"AU6": 0.3, "AU12": 0.4},
    "sad": {"AU1": 0.3, "AU15": 0.3, "AU17": 0.3},
    "angry": {"AU4": 0.4, "AU17": 0.2},
    "surprised": {"AU1": 0.4, "AU2": 0.4, "AU26": 0.4},
    "fearful": {"AU1": 0.3, "AU2": 0.3, "AU20": 0.3},
    "disgusted": {"AU4": 0.3, "AU17": 0.2},
    "contempt": {"AU4": 0.2, "AU12": 0.2},  # asymmetric lip pull — approximated
    "neutral": {},
}

MICROEXPRESSION_THRESHOLD_MS = 200.0


@dataclass
class VisualFeatures:
    """Container for all extracted facial and visual features."""

    # Expression classification
    expression_label: str = "neutral"
    expression_confidence: float = 0.0

    # Facial Action Unit scores (0.0–1.0 each)
    au_scores: Dict[str, float] = field(default_factory=dict)

    # Eye analysis
    eye_openness: float = 0.0      # Eye Aspect Ratio (EAR)
    blink_rate: float = 0.0        # blinks per second
    gaze_direction: str = "center"  # left / right / up / down / center

    # Microexpression detection
    microexpression_detected: bool = False
    microexpression_au: str = ""

    # Metadata
    landmark_count: int = 0
    frame_count: int = 0
    confidence: float = 0.0
    available: bool = True


class VisualAnalyzer:
    """
    Analyzes video frames using MediaPipe FaceMesh to extract facial action units
    and detect emotional expression patterns.

    Returns None / empty VisualFeatures when MediaPipe is unavailable.
    """

    def __init__(
        self,
        min_face_confidence: float = 0.5,
        microexpression_threshold_ms: float = MICROEXPRESSION_THRESHOLD_MS,
    ) -> None:
        self.min_face_confidence = min_face_confidence
        self.microexpression_threshold_ms = microexpression_threshold_ms
        self._mp_available = self._check_mediapipe()
        self._face_mesh = None
        self._prev_au_scores: Optional[Dict[str, float]] = None
        self._prev_au_timestamp_ms: float = 0.0

    def is_available(self) -> bool:
        """Return True if MediaPipe FaceMesh is installed and functional."""
        return self._mp_available

    def analyze_image(self, path: str) -> Optional[VisualFeatures]:
        """Analyze a single image file and return VisualFeatures."""
        if not self._mp_available:
            return self._unavailable_features()
        try:
            import cv2

            frame = cv2.imread(path)
            if frame is None:
                logger.warning("Could not read image: %s", path)
                return self._unavailable_features()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.analyze_video_frame(frame_rgb)
        except Exception as exc:
            logger.error("Image analysis failed for %s: %s", path, exc)
            return self._unavailable_features()

    def analyze_video_frame(self, frame: np.ndarray) -> Optional[VisualFeatures]:
        """
        Analyze a single RGB numpy array frame.
        Returns VisualFeatures or None if face not detected.
        """
        if not self._mp_available:
            return self._unavailable_features()
        try:
            import mediapipe as mp

            mesh = self._get_face_mesh()
            results = mesh.process(frame)
            if not results.multi_face_landmarks:
                return VisualFeatures(
                    available=True,
                    landmark_count=0,
                    confidence=0.0,
                    expression_label="no_face_detected",
                )

            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            landmark_count = len(landmarks)

            # Convert landmarks to numpy array (x, y, z)
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

            au_scores = self._compute_action_units(lm_array)
            expression_label, expression_confidence = self._classify_expression(au_scores)
            eye_openness = self._compute_ear(lm_array)
            gaze_direction = self._estimate_gaze(lm_array)

            # Microexpression detection
            micro_detected = False
            micro_au = ""
            current_time_ms = time.time() * 1000
            if self._prev_au_scores is not None:
                elapsed = current_time_ms - self._prev_au_timestamp_ms
                micro_detected, micro_au = self._detect_microexpression(
                    self._prev_au_scores, au_scores, elapsed
                )
            self._prev_au_scores = au_scores.copy()
            self._prev_au_timestamp_ms = current_time_ms

            return VisualFeatures(
                expression_label=expression_label,
                expression_confidence=expression_confidence,
                au_scores=au_scores,
                eye_openness=eye_openness,
                gaze_direction=gaze_direction,
                microexpression_detected=micro_detected,
                microexpression_au=micro_au,
                landmark_count=landmark_count,
                frame_count=1,
                confidence=float(min(1.0, landmark_count / 300.0)),
                available=True,
            )

        except Exception as exc:
            logger.error("Frame analysis failed: %s", exc)
            return self._unavailable_features()

    def analyze_video_file(self, path: str) -> List[VisualFeatures]:
        """
        Analyze a video file frame by frame.
        Returns a list of VisualFeatures (one per sampled frame).
        Samples at 5 FPS to reduce computation.
        """
        if not self._mp_available:
            return [self._unavailable_features()]
        results_list = []
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.warning("Could not open video file: %s", path)
                return [self._unavailable_features()]

            fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = 5.0
            frame_skip = max(1, int(fps / target_fps)) if fps > 0 else 3

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    vf = self.analyze_video_frame(frame_rgb)
                    if vf is not None:
                        vf.frame_count = frame_idx
                        results_list.append(vf)
                frame_idx += 1
            cap.release()

        except Exception as exc:
            logger.error("Video file analysis failed for %s: %s", path, exc)
            return [self._unavailable_features()]

        return results_list if results_list else [self._unavailable_features()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_mediapipe(self) -> bool:
        """Check if mediapipe is installed without crashing."""
        try:
            import mediapipe  # noqa: F401
            import cv2  # noqa: F401
            return True
        except ImportError:
            logger.info("MediaPipe or OpenCV not available — visual analysis disabled")
            return False

    def _get_face_mesh(self):
        """Lazily initialize MediaPipe FaceMesh."""
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_face_confidence,
                min_tracking_confidence=0.5,
            )
        return self._face_mesh

    def _compute_action_units(self, lm: np.ndarray) -> Dict[str, float]:
        """
        Compute AU scores from landmark positions.
        Each AU score is a normalized float 0.0–1.0.
        """
        au_scores: Dict[str, float] = {au: 0.0 for au in AU_DESCRIPTIONS}

        try:
            # AU1 — Inner Brow Raise: inner brow Y vs nose bridge Y
            left_inner_brow_y = lm[LANDMARK_INDICES["left_inner_brow"]][1]
            right_inner_brow_y = lm[LANDMARK_INDICES["right_inner_brow"]][1]
            nose_bridge_y = lm[LANDMARK_INDICES["nose_bridge"]][1]
            brow_raise = max(0.0, nose_bridge_y - (left_inner_brow_y + right_inner_brow_y) / 2)
            au_scores["AU1"] = float(min(1.0, brow_raise * 10.0))

            # AU2 — Outer Brow Raise
            left_outer_brow_y = lm[LANDMARK_INDICES["left_outer_brow"]][1]
            right_outer_brow_y = lm[LANDMARK_INDICES["right_outer_brow"]][1]
            outer_raise = max(0.0, nose_bridge_y - (left_outer_brow_y + right_outer_brow_y) / 2)
            au_scores["AU2"] = float(min(1.0, outer_raise * 8.0))

            # AU4 — Brow Lowerer: brows converge toward center
            brow_distance = abs(
                lm[LANDMARK_INDICES["left_inner_brow"]][0]
                - lm[LANDMARK_INDICES["right_inner_brow"]][0]
            )
            au_scores["AU4"] = float(min(1.0, max(0.0, 0.3 - brow_distance) * 5.0))

            # AU6/AU12 — Duchenne smile: cheek raise + lip corner pull
            lip_left = lm[LANDMARK_INDICES["lip_left_corner"]]
            lip_right = lm[LANDMARK_INDICES["lip_right_corner"]]
            lip_corner_y_avg = (lip_left[1] + lip_right[1]) / 2
            lip_center_y = lm[LANDMARK_INDICES["upper_lip_top"]][1]
            smile_indicator = max(0.0, lip_center_y - lip_corner_y_avg)
            au_scores["AU12"] = float(min(1.0, smile_indicator * 15.0))
            left_cheek_y = lm[LANDMARK_INDICES["left_cheek"]][1]
            right_cheek_y = lm[LANDMARK_INDICES["right_cheek"]][1]
            cheek_raise = max(0.0, nose_bridge_y - (left_cheek_y + right_cheek_y) / 2)
            au_scores["AU6"] = float(min(1.0, cheek_raise * 8.0))

            # AU15/AU17 — Lip corner depress / chin raise (sadness)
            lip_depress = max(0.0, (lip_left[1] + lip_right[1]) / 2 - lip_center_y - 0.01)
            au_scores["AU15"] = float(min(1.0, lip_depress * 20.0))
            chin_y = lm[LANDMARK_INDICES["chin"]][1]
            lower_lip_y = lm[LANDMARK_INDICES["lower_lip_bottom"]][1]
            chin_raise = max(0.0, lower_lip_y - chin_y)
            au_scores["AU17"] = float(min(1.0, chin_raise * 10.0))

            # AU20 — Lip Stretcher (fear)
            lip_width = abs(lip_left[0] - lip_right[0])
            face_width = abs(lm[0][0] - lm[16][0]) if len(lm) > 16 else 0.3
            lip_stretch = lip_width / face_width if face_width > 0 else 0.0
            au_scores["AU20"] = float(min(1.0, max(0.0, lip_stretch - 0.4) * 5.0))

            # AU26 — Jaw Drop
            upper_lip_y = lm[LANDMARK_INDICES["upper_lip_top"]][1]
            lower_lip_y = lm[LANDMARK_INDICES["lower_lip_bottom"]][1]
            mouth_open = max(0.0, lower_lip_y - upper_lip_y - 0.01)
            au_scores["AU26"] = float(min(1.0, mouth_open * 20.0))

        except (IndexError, ZeroDivisionError) as exc:
            logger.debug("AU computation error: %s", exc)

        return au_scores

    def _classify_expression(
        self, au_scores: Dict[str, float]
    ) -> tuple[str, float]:
        """Classify expression from AU scores using maximum likelihood matching."""
        best_label = "neutral"
        best_score = 0.0

        for expression, required_aus in EXPRESSION_AU_MAP.items():
            if not required_aus:
                continue
            score = 0.0
            for au, threshold in required_aus.items():
                if au in au_scores and au_scores[au] >= threshold:
                    score += au_scores[au]
            score /= max(1, len(required_aus))
            if score > best_score:
                best_score = score
                best_label = expression

        return best_label, float(min(1.0, best_score))

    def _compute_ear(self, lm: np.ndarray) -> float:
        """Compute Eye Aspect Ratio (EAR) as a measure of eye openness."""
        try:
            # Left eye EAR
            left_top = lm[LANDMARK_INDICES["left_eye_top"]]
            left_bottom = lm[LANDMARK_INDICES["left_eye_bottom"]]
            left_left = lm[LANDMARK_INDICES["left_eye_left"]]
            left_right = lm[LANDMARK_INDICES["left_eye_right"]]
            left_vertical = np.linalg.norm(left_top - left_bottom)
            left_horizontal = np.linalg.norm(left_left - left_right)
            left_ear = left_vertical / (left_horizontal + 1e-6)

            # Right eye EAR
            right_top = lm[LANDMARK_INDICES["right_eye_top"]]
            right_bottom = lm[LANDMARK_INDICES["right_eye_bottom"]]
            right_left = lm[LANDMARK_INDICES["right_eye_left"]]
            right_right = lm[LANDMARK_INDICES["right_eye_right"]]
            right_vertical = np.linalg.norm(right_top - right_bottom)
            right_horizontal = np.linalg.norm(right_left - right_right)
            right_ear = right_vertical / (right_horizontal + 1e-6)

            return float((left_ear + right_ear) / 2.0)
        except Exception:
            return 0.0

    def _estimate_gaze(self, lm: np.ndarray) -> str:
        """
        Estimate gaze direction from eye landmark positions.
        Returns: left / right / up / down / center
        """
        try:
            # Use iris position relative to eye corners (MediaPipe 478-point model)
            # Fallback: use eye midpoint relative to face center
            left_eye_center_x = (
                lm[LANDMARK_INDICES["left_eye_left"]][0]
                + lm[LANDMARK_INDICES["left_eye_right"]][0]
            ) / 2
            face_center_x = lm[LANDMARK_INDICES["nose_bridge"]][0]
            offset = left_eye_center_x - face_center_x
            if offset < -0.02:
                return "left"
            elif offset > 0.02:
                return "right"
            else:
                return "center"
        except Exception:
            return "center"

    def _detect_microexpression(
        self,
        prev_aus: Dict[str, float],
        curr_aus: Dict[str, float],
        elapsed_ms: float,
    ) -> tuple[bool, str]:
        """
        Detect microexpressions: significant AU changes within <200ms.
        Microexpressions are involuntary, brief emotional leaks.
        """
        if elapsed_ms <= 0 or elapsed_ms > self.microexpression_threshold_ms:
            return False, ""

        # Check for significant rapid change in any AU
        MAX_CHANGE_THRESHOLD = 0.4
        for au in prev_aus:
            if au in curr_aus:
                change = abs(curr_aus[au] - prev_aus[au])
                if change >= MAX_CHANGE_THRESHOLD:
                    return True, au
        return False, ""

    def _unavailable_features(self) -> VisualFeatures:
        """Return empty VisualFeatures indicating MediaPipe is unavailable."""
        return VisualFeatures(available=False)

    def get_aggregate_features(self, frame_features: List[VisualFeatures]) -> VisualFeatures:
        """
        Aggregate per-frame VisualFeatures into a single representative summary.
        Used when analyzing a full video file.
        """
        if not frame_features:
            return self._unavailable_features()

        valid_frames = [f for f in frame_features if f.available and f.landmark_count > 0]
        if not valid_frames:
            return self._unavailable_features()

        # Most common expression label (mode)
        from collections import Counter
        expression_counts = Counter(f.expression_label for f in valid_frames)
        dominant_expression = expression_counts.most_common(1)[0][0]
        dominant_confidence = (
            expression_counts[dominant_expression] / len(valid_frames)
        )

        # Average AU scores
        avg_aus: Dict[str, float] = {}
        for au in AU_DESCRIPTIONS:
            scores = [f.au_scores.get(au, 0.0) for f in valid_frames]
            avg_aus[au] = float(np.mean(scores))

        # Average eye openness
        avg_ear = float(np.mean([f.eye_openness for f in valid_frames]))

        # Blink rate estimation
        blink_count = sum(
            1 for f in valid_frames
            if f.eye_openness < 0.15  # closed eye threshold
        )
        duration_sec = len(valid_frames) / 5.0  # assuming 5 FPS sampling
        blink_rate = blink_count / duration_sec if duration_sec > 0 else 0.0

        # Any microexpression
        micro_detected = any(f.microexpression_detected for f in valid_frames)
        micro_aus = [f.microexpression_au for f in valid_frames if f.microexpression_detected]
        micro_au = micro_aus[0] if micro_aus else ""

        return VisualFeatures(
            expression_label=dominant_expression,
            expression_confidence=dominant_confidence,
            au_scores=avg_aus,
            eye_openness=avg_ear,
            blink_rate=blink_rate,
            gaze_direction=valid_frames[-1].gaze_direction if valid_frames else "center",
            microexpression_detected=micro_detected,
            microexpression_au=micro_au,
            landmark_count=int(np.mean([f.landmark_count for f in valid_frames])),
            frame_count=len(frame_features),
            confidence=float(np.mean([f.confidence for f in valid_frames])),
            available=True,
        )
