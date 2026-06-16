"""
audio_analyzer.py — Prosodic and spectral audio feature extraction.

Extracts MFCC (40 coefficients + delta + delta2), chroma, spectral features,
F0 contour (female voice range 75–400 Hz), speech rate, pause patterns,
and optional wav2vec2 768-dim embeddings for behavioral analysis.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Female voice F0 range constants
F0_FMIN = 75.0   # Hz — lower bound for female fundamental frequency
F0_FMAX = 400.0  # Hz — upper bound for female fundamental frequency

# Audio quality thresholds
MIN_RMS_ENERGY = 0.001
MIN_DURATION_SEC = 0.5
SPECTRAL_CENTROID_MIN = 50.0   # Hz
SPECTRAL_CENTROID_MAX = 8000.0  # Hz

# Vocal cue detection thresholds
HEDGING_PAUSE_THRESHOLD_SEC = 0.3   # pauses longer than this are "hedging pauses"
VOICE_RISE_WINDOW_SEC = 0.2         # last 200ms of utterance for rising tone detection
EMOTIONAL_STRESS_F0_STD_THRESHOLD = 30.0  # Hz std deviation indicating stress


@dataclass
class ProsodicFeatures:
    """Container for all extracted prosodic and spectral audio features."""

    # MFCC features (40 coefficients + delta + delta2)
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(40))
    mfcc_std: np.ndarray = field(default_factory=lambda: np.zeros(40))
    mfcc_delta_mean: np.ndarray = field(default_factory=lambda: np.zeros(40))
    mfcc_delta2_mean: np.ndarray = field(default_factory=lambda: np.zeros(40))

    # Chroma features (12 bins)
    chroma_mean: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # Spectral features
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    spectral_rolloff: float = 0.0

    # Zero-crossing rate and energy
    zcr: float = 0.0
    rms: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0

    # Tempo
    tempo: float = 0.0

    # F0 (fundamental frequency / pitch) features
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    f0_values: np.ndarray = field(default_factory=lambda: np.array([]))

    # Prosodic / speech rate features
    speech_rate: float = 0.0       # estimated syllables per second
    pause_count: int = 0
    pause_ratio: float = 0.0       # fraction of total duration that is pause
    duration_seconds: float = 0.0

    # Vocal cue detection results
    vocal_cues: dict = field(default_factory=dict)
    # Keys: has_hedging_pauses, has_voice_rise, has_emotional_stress

    # Optional wav2vec2 embedding
    has_wav2vec2: bool = False
    wav2vec2_embedding: Optional[np.ndarray] = None

    # Metadata
    sample_rate: int = 16000
    audio_valid: bool = True
    validation_message: str = ""


class AudioAnalyzer:
    """
    Extracts prosodic, spectral, and deep speech features from audio files
    or live microphone recordings.

    All analysis is optimized for female voice characteristics (F0 range 75–400 Hz).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mfcc_coefficients: int = 40,
        chroma_bins: int = 12,
        use_wav2vec2: bool = True,
        hf_model_manager=None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mfcc_coefficients = mfcc_coefficients
        self.chroma_bins = chroma_bins
        self.use_wav2vec2 = use_wav2vec2
        self._hf_manager = hf_model_manager

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_file(self, path: str) -> ProsodicFeatures:
        """Load audio from file path and extract all prosodic features."""
        if not self.validate_audio(path):
            return ProsodicFeatures(
                audio_valid=False,
                validation_message=f"Audio file failed quality gates: {path}",
            )
        try:
            y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return self._extract_features(y, sr)
        except Exception as exc:
            logger.error("Failed to analyze audio file %s: %s", path, exc)
            return ProsodicFeatures(audio_valid=False, validation_message=str(exc))

    def analyze_microphone(self, duration: int = 30) -> ProsodicFeatures:
        """
        Record audio from the default microphone for `duration` seconds,
        then extract all prosodic features.
        """
        try:
            import sounddevice as sd  # lazy import — not required for file analysis

            logger.info("Recording %d seconds from microphone…", duration)
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            y = recording.flatten()
            return self._extract_features(y, self.sample_rate)
        except ImportError:
            logger.warning("sounddevice not installed — cannot record from microphone")
            return ProsodicFeatures(
                audio_valid=False,
                validation_message="sounddevice not installed",
            )
        except Exception as exc:
            logger.error("Microphone recording failed: %s", exc)
            return ProsodicFeatures(audio_valid=False, validation_message=str(exc))

    def validate_audio(self, path: str) -> bool:
        """
        Three-gate audio quality check:
        1. RMS energy > MIN_RMS_ENERGY (not silence)
        2. Duration > MIN_DURATION_SEC (long enough to analyse)
        3. Spectral centroid in [SPECTRAL_CENTROID_MIN, SPECTRAL_CENTROID_MAX] (valid speech)
        """
        if not os.path.exists(path):
            logger.warning("Audio file not found: %s", path)
            return False
        try:
            y, sr = librosa.load(path, sr=None, mono=True, duration=60.0)

            # Gate 1: RMS energy
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms < MIN_RMS_ENERGY:
                logger.warning("Audio rejected: RMS %.4f < %.4f (silent)", rms, MIN_RMS_ENERGY)
                return False

            # Gate 2: Duration
            duration = len(y) / sr
            if duration < MIN_DURATION_SEC:
                logger.warning(
                    "Audio rejected: duration %.2fs < %.2fs", duration, MIN_DURATION_SEC
                )
                return False

            # Gate 3: Spectral centroid
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            if not (SPECTRAL_CENTROID_MIN <= centroid <= SPECTRAL_CENTROID_MAX):
                logger.warning(
                    "Audio rejected: spectral centroid %.1f Hz outside [%.0f, %.0f] Hz",
                    centroid,
                    SPECTRAL_CENTROID_MIN,
                    SPECTRAL_CENTROID_MAX,
                )
                return False

            return True
        except Exception as exc:
            logger.error("Audio validation error for %s: %s", path, exc)
            return False

    def to_feature_vector(self, features: ProsodicFeatures) -> np.ndarray:
        """
        Concatenate all handcrafted features (and optional wav2vec2 embedding)
        into a single flat float32 numpy array suitable for downstream classifiers.
        """
        parts = [
            features.mfcc_mean,         # 40
            features.mfcc_std,          # 40
            features.mfcc_delta_mean,   # 40
            features.mfcc_delta2_mean,  # 40
            features.chroma_mean,       # 12
            np.array([
                features.spectral_centroid,
                features.spectral_bandwidth,
                features.spectral_rolloff,
                features.zcr,
                features.rms,
                features.energy_mean,
                features.energy_std,
                features.tempo,
                features.f0_mean,
                features.f0_std,
                features.f0_range,
                features.speech_rate,
                float(features.pause_count),
                features.pause_ratio,
                features.duration_seconds,
            ]),  # 15
        ]
        vec = np.concatenate([p.astype(np.float32) for p in parts])

        if features.has_wav2vec2 and features.wav2vec2_embedding is not None:
            vec = np.concatenate([vec, features.wav2vec2_embedding.astype(np.float32)])

        return vec

    # ------------------------------------------------------------------
    # Private feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, y: np.ndarray, sr: int) -> ProsodicFeatures:
        """Core feature extraction pipeline on a loaded audio array."""
        features = ProsodicFeatures(sample_rate=sr)
        features.duration_seconds = float(len(y) / sr)

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.mfcc_coefficients)
        features.mfcc_mean = np.mean(mfcc, axis=1)
        features.mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        features.mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

        # --- Chroma ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.chroma_bins)
        features.chroma_mean = np.mean(chroma, axis=1)

        # --- Spectral features ---
        features.spectral_centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        )
        features.spectral_bandwidth = float(
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        )
        features.spectral_rolloff = float(
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        )

        # --- ZCR and RMS ---
        zcr = librosa.feature.zero_crossing_rate(y)
        features.zcr = float(np.mean(zcr))
        rms_frames = librosa.feature.rms(y=y)
        features.rms = float(np.mean(rms_frames))
        features.energy_mean = float(np.mean(y ** 2))
        features.energy_std = float(np.std(y ** 2))

        # --- Tempo ---
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        except Exception:
            features.tempo = 0.0

        # --- F0 (pitch) ---
        features = self._extract_f0(y, sr, features)

        # --- Speech rate and pauses ---
        features = self._extract_speech_rate(y, sr, features)

        # --- Vocal cues ---
        features.vocal_cues = {
            "has_hedging_pauses": self._detect_hedging_pauses(y, sr),
            "has_voice_rise": self._detect_voice_rise(features.f0_values),
            "has_emotional_stress": self._detect_emotional_stress(features.f0_values),
        }

        # --- Optional wav2vec2 embedding ---
        if self.use_wav2vec2 and self._hf_manager is not None:
            try:
                embedding = self._hf_manager.extract_wav2vec2(y, sr)
                if embedding is not None:
                    features.has_wav2vec2 = True
                    features.wav2vec2_embedding = embedding
            except Exception as exc:
                logger.debug("wav2vec2 extraction skipped: %s", exc)

        return features

    def _extract_f0(self, y: np.ndarray, sr: int, features: ProsodicFeatures) -> ProsodicFeatures:
        """Extract fundamental frequency (F0) using pyin algorithm."""
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=F0_FMIN,
                fmax=F0_FMAX,
                sr=sr,
            )
            # Keep only voiced frames
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
            features.f0_values = voiced_f0

            if len(voiced_f0) > 0:
                features.f0_mean = float(np.mean(voiced_f0))
                features.f0_std = float(np.std(voiced_f0))
                features.f0_range = float(np.max(voiced_f0) - np.min(voiced_f0))
            else:
                features.f0_mean = 0.0
                features.f0_std = 0.0
                features.f0_range = 0.0
        except Exception as exc:
            logger.debug("F0 extraction failed: %s", exc)
            features.f0_values = np.array([])

        return features

    def _extract_speech_rate(
        self, y: np.ndarray, sr: int, features: ProsodicFeatures
    ) -> ProsodicFeatures:
        """Estimate speech rate and pause statistics from energy contour."""
        try:
            # Frame-level RMS for pause detection
            hop_length = 512
            frame_length = 2048
            rms_frames = librosa.feature.rms(
                y=y, frame_length=frame_length, hop_length=hop_length
            )[0]
            frame_duration = hop_length / sr  # seconds per frame

            # Threshold: frames with RMS < 10% of max are pauses
            threshold = 0.1 * np.max(rms_frames) if np.max(rms_frames) > 0 else MIN_RMS_ENERGY
            is_pause = rms_frames < threshold

            # Count pause segments
            pause_segments = 0
            in_pause = False
            pause_frames = 0
            pause_segment_durations = []
            current_pause_length = 0

            for i, p in enumerate(is_pause):
                if p:
                    if not in_pause:
                        in_pause = True
                        current_pause_length = 1
                    else:
                        current_pause_length += 1
                    pause_frames += 1
                else:
                    if in_pause:
                        pause_duration = current_pause_length * frame_duration
                        if pause_duration >= HEDGING_PAUSE_THRESHOLD_SEC * 0.5:
                            pause_segment_durations.append(pause_duration)
                            pause_segments += 1
                        in_pause = False
                        current_pause_length = 0

            total_frames = len(rms_frames)
            features.pause_count = pause_segments
            features.pause_ratio = float(pause_frames / total_frames) if total_frames > 0 else 0.0

            # Estimate syllable rate from onset envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sr, hop_length=hop_length
            )
            speech_duration = features.duration_seconds * (1.0 - features.pause_ratio)
            if speech_duration > 0:
                features.speech_rate = len(onset_frames) / speech_duration
            else:
                features.speech_rate = 0.0

        except Exception as exc:
            logger.debug("Speech rate extraction failed: %s", exc)

        return features

    def _detect_hedging_pauses(self, y: np.ndarray, sr: int) -> bool:
        """
        Detect pauses longer than HEDGING_PAUSE_THRESHOLD_SEC (0.3s).
        These are associated with uncertainty and indirect communication.
        """
        try:
            hop_length = 512
            rms_frames = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            frame_duration = hop_length / sr
            threshold = 0.1 * np.max(rms_frames) if np.max(rms_frames) > 0 else MIN_RMS_ENERGY

            in_pause = False
            current_pause = 0.0
            for is_low in rms_frames < threshold:
                if is_low:
                    current_pause += frame_duration
                    in_pause = True
                else:
                    if in_pause and current_pause >= HEDGING_PAUSE_THRESHOLD_SEC:
                        return True
                    current_pause = 0.0
                    in_pause = False

            return False
        except Exception:
            return False

    def _detect_voice_rise(self, f0_values: np.ndarray) -> bool:
        """
        Detect rising F0 pattern in the final portion of an utterance.
        Rising intonation at statement end indicates a question or uncertainty
        (common in anxious attachment and indirect communication).
        """
        if len(f0_values) < 10:
            return False
        try:
            # Take the last 20% of voiced F0 values
            tail_n = max(3, len(f0_values) // 5)
            tail = f0_values[-tail_n:]
            # Linear regression slope > 5 Hz/frame indicates rising
            x = np.arange(len(tail))
            if len(x) < 2:
                return False
            slope = np.polyfit(x, tail, 1)[0]
            return float(slope) > 5.0
        except Exception:
            return False

    def _detect_emotional_stress(self, f0_values: np.ndarray) -> bool:
        """
        Detect high F0 variance as a proxy for emotional stress/arousal.
        High variance (> EMOTIONAL_STRESS_F0_STD_THRESHOLD Hz) indicates
        emotional volatility associated with high-arousal states.
        """
        if len(f0_values) < 5:
            return False
        return float(np.std(f0_values)) > EMOTIONAL_STRESS_F0_STD_THRESHOLD
