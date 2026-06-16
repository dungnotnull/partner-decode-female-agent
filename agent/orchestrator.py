"""
orchestrator.py — PartnerDecodeOrchestrator: the core agent decision loop.

Wires together AudioAnalyzer, VisualAnalyzer, BehaviorClassifier,
InterpretationEngine, MemoryManager, and KnowledgeUpdater into a
coherent multimodal analysis pipeline with quality gates at each stage.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality Gate Thresholds (hard-coded per PROJECT-detail.md)
# ---------------------------------------------------------------------------
EMOTION_CONFIDENCE_THRESHOLD = 0.3      # Gate 2: if all emotion scores < 0.3, label "uncertain"
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.2  # Gate 3: if overall confidence < 0.2, warn
COUNSELING_DISTRESS_THRESHOLD = 0.8    # Gate 4: distress > 0.8 → counseling
COUNSELING_HORSEMAN_THRESHOLD = 0.7    # Gate 4: any horseman > 0.7 → counseling


class PartnerDecodeOrchestrator:
    """
    Central coordinator for the partner-decode-female-agent.

    Implements lazy module initialization so heavy models are only loaded
    on first use rather than at startup.

    Quality Gates:
      1. Audio Quality Gate: RMS > 0.001, duration > 0.5s, spectral centroid 50–8000 Hz
      2. Emotion Confidence Gate: all emotion scores < 0.3 → label "uncertain"
      3. Classification Confidence Gate: overall confidence < 0.2 → add warning
      4. Safety Gate: distress > 0.8 OR horseman > 0.7 → counseling_flag
      5. LLM Output Gate: JSON parse must succeed OR fallback
      6. Harm Prevention Gate: scan for prohibited content
      7. Privacy Gate: PRIVACY_MODE=true → Ollama only (enforced in LLMClient)
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config = config or {}
        # Lazy-loaded module references
        self._audio_analyzer: Optional[Any] = None
        self._visual_analyzer: Optional[Any] = None
        self._behavior_classifier: Optional[Any] = None
        self._interpretation_engine: Optional[Any] = None
        self._memory: Optional[Any] = None
        self._knowledge_updater: Optional[Any] = None
        self._llm_client: Optional[Any] = None
        self._hf_manager: Optional[Any] = None
        self._scheduler = None

        # Quality gate failure log
        self._gate_failures: list = []

        # Prometheus counters
        self._prom_sessions_total = 0
        self._prom_errors_total = 0
        self._prom_cost_usd_total = 0.0
        self._prom_latency_seconds_total = 0.0

    # ------------------------------------------------------------------
    # Lazy Module Initializers
    # ------------------------------------------------------------------

    def _get_hf_manager(self):
        if self._hf_manager is None:
            from tools.hf_model_manager import HFModelManager
            self._hf_manager = HFModelManager.get_instance(
                cache_dir=self._config.get("hf_models", {}).get("cache_dir", "./models")
            )
        return self._hf_manager

    def _get_llm_client(self):
        if self._llm_client is None:
            from tools.llm_client import LLMClient
            self._llm_client = LLMClient(
                memory_manager=self._get_memory(),
                primary_provider=self._config.get("llm", {}).get("primary_provider", "claude"),
                primary_model=self._config.get("llm", {}).get("primary_model", "claude-opus-4-8"),
                fallback_provider=self._config.get("llm", {}).get("fallback_provider", "openai"),
                fallback_model=self._config.get("llm", {}).get("fallback_model", "gpt-4o"),
                offline_model=self._config.get("llm", {}).get("offline_model", "llama3"),
            )
        return self._llm_client

    def _get_memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            db_path = self._config.get("memory", {}).get("db_path", "./data/partner_decode.db")
            self._memory = MemoryManager(db_path=db_path)
        return self._memory

    def _get_audio_analyzer(self):
        if self._audio_analyzer is None:
            from agent.modules.audio_analyzer import AudioAnalyzer
            feat_cfg = self._config.get("feature_extraction", {})
            self._audio_analyzer = AudioAnalyzer(
                sample_rate=self._config.get("audio", {}).get("sample_rate", 16000),
                mfcc_coefficients=feat_cfg.get("mfcc_coefficients", 40),
                chroma_bins=feat_cfg.get("chroma_bins", 12),
                use_wav2vec2=feat_cfg.get("use_wav2vec2", True),
                hf_model_manager=self._get_hf_manager(),
            )
        return self._audio_analyzer

    def _get_visual_analyzer(self):
        if self._visual_analyzer is None:
            from agent.modules.visual_analyzer import VisualAnalyzer
            vis_cfg = self._config.get("visual_analysis", {})
            self._visual_analyzer = VisualAnalyzer(
                min_face_confidence=vis_cfg.get("min_face_confidence", 0.5),
                microexpression_threshold_ms=vis_cfg.get("microexpression_threshold_ms", 200.0),
            )
        return self._visual_analyzer

    def _get_classifier(self):
        if self._behavior_classifier is None:
            from agent.modules.behavior_classifier import BehaviorClassifier
            self._behavior_classifier = BehaviorClassifier(
                hf_model_manager=self._get_hf_manager()
            )
        return self._behavior_classifier

    def _get_interpreter(self):
        if self._interpretation_engine is None:
            from agent.modules.interpretation_engine import InterpretationEngine
            self._interpretation_engine = InterpretationEngine(
                llm_client=self._get_llm_client(),
                hf_model_manager=self._get_hf_manager(),
            )
        return self._interpretation_engine

    def _get_knowledge_updater(self):
        if self._knowledge_updater is None:
            from tools.knowledge_updater import KnowledgeUpdater
            ku_cfg = self._config.get("knowledge_updater", {})
            self._knowledge_updater = KnowledgeUpdater(
                memory_manager=self._get_memory(),
                max_papers_per_run=ku_cfg.get("max_papers_per_run", 50),
            )
        return self._knowledge_updater

    # ------------------------------------------------------------------
    # Quality Gate Checks
    # ------------------------------------------------------------------

    def _check_emotion_confidence_gate(self, emotion_scores: Dict) -> Dict:
        """
        Quality Gate 2: If all emotion classification scores are below
        threshold, label as "uncertain" rather than forcing a label.
        """
        if not emotion_scores:
            return emotion_scores

        max_score = max(emotion_scores.values()) if emotion_scores else 0.0
        if max_score < EMOTION_CONFIDENCE_THRESHOLD:
            logger.info(
                "Gate 2 (Emotion Confidence): max emotion score %.3f < %.3f — labeling as uncertain",
                max_score, EMOTION_CONFIDENCE_THRESHOLD,
            )
            emotion_scores["uncertain"] = 1.0 - max_score
            self._gate_failures.append("emotion_confidence_low")
        return emotion_scores

    def _check_classification_confidence_gate(self, classification_result) -> dict:
        """
        Quality Gate 3: If BehaviorClassifier overall confidence is below
        threshold, add a warning flag. The analysis still proceeds, but
        the interpretation engine is informed that signal is weak.
        """
        warning = None
        if classification_result.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
            warning = (
                f"Classification confidence ({classification_result.confidence:.2f}) is below "
                f"threshold ({CLASSIFICATION_CONFIDENCE_THRESHOLD}). Results may be unreliable."
            )
            logger.warning("Gate 3 (Classification Confidence): %s", warning)
            self._gate_failures.append("classification_confidence_low")
        return {"warning": warning, "classification": classification_result}

    def _check_safety_gate(self, classification_result) -> None:
        """
        Quality Gate 4: Hard-coded safety thresholds.
        If distress_score > 0.8 or any Horseman > 0.7, ensure
        counseling_flag is set. This gate is also enforced in
        BehaviorClassifier, but we verify again here.
        """
        max_horseman = max(classification_result.gottman_horsemen.values()) if classification_result.gottman_horsemen else 0.0
        if classification_result.overall_distress_score > COUNSELING_DISTRESS_THRESHOLD:
            logger.warning(
                "Gate 4 (Safety): distress_score=%.3f exceeds threshold=%.3f — counseling recommended",
                classification_result.overall_distress_score, COUNSELING_DISTRESS_THRESHOLD,
            )
            if not classification_result.needs_counseling_flag:
                classification_result.needs_counseling_flag = True

        if max_horseman > COUNSELING_HORSEMAN_THRESHOLD:
            logger.warning(
                "Gate 4 (Safety): max_horseman=%.3f exceeds threshold=%.3f — counseling recommended",
                max_horseman, COUNSELING_HORSEMAN_THRESHOLD,
            )
            if not classification_result.needs_counseling_flag:
                classification_result.needs_counseling_flag = True

    def _log_gate_error(self, gate_name: str, message: str) -> None:
        """Log a quality gate failure."""
        logger.error("Quality Gate FAILED — %s: %s", gate_name, message)
        self._gate_failures.append(f"{gate_name}: {message}")

    # ------------------------------------------------------------------
    # Core Analysis Methods
    # ------------------------------------------------------------------

    def analyze_audio(
        self, audio_path: str, session_id: Optional[str] = None
    ) -> Dict:
        """
        Full audio analysis pipeline:
        validate → transcribe → extract features → classify → gate check → interpret → save
        """
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        self._gate_failures = []

        try:
            self._prom_sessions_total += 1
            audio_analyzer = self._get_audio_analyzer()
            classifier = self._get_classifier()
            interpreter = self._get_interpreter()
            memory = self._get_memory()

            # Gate 1: Audio Quality Gate
            if not audio_analyzer.validate_audio(audio_path):
                self._log_gate_error("audio_quality", f"Audio failed validation: {audio_path}")
                return {
                    "session_id": session_id,
                    "error": "Audio failed quality gates (RMS/duration/spectral_centroid)",
                    "status": "validation_failed",
                }

            # Transcribe
            text = ""
            try:
                hf = self._get_hf_manager()
                transcription = hf.transcribe(audio_path)
                text = transcription.get("text", "") if isinstance(transcription, dict) else str(transcription)
            except Exception as exc:
                logger.warning("Transcription failed: %s — proceeding without text", exc)

            # Extract prosodic features
            audio_features = audio_analyzer.analyze_file(audio_path)

            # Classify behavior
            classification = classifier.classify(audio_features, text, None)

            # Gate 2: Emotion confidence check
            if classification.emotion_scores:
                classification.emotion_scores = self._check_emotion_confidence_gate(
                    classification.emotion_scores
                )

            # Gate 3: Classification confidence check
            gate3_result = self._check_classification_confidence_gate(classification)
            classification = gate3_result["classification"]

            # Gate 4: Safety gate
            self._check_safety_gate(classification)

            # Interpret
            interpretation = interpreter.interpret(
                classification, audio_features, text, None
            )

            # Gate 5: LLM output is already handled in InterpretationEngine._parse_llm_response
            # Gate 6: Harm prevention is already handled in InterpretationEngine._scan_for_harm

            # Persist
            latency_ms = (time.time() - start_time) * 1000
            self._prom_latency_seconds_total += latency_ms / 1000.0
            memory.save_session(
                session_id=session_id,
                audio_path=audio_path,
                text_input=text,
                has_video=False,
                classification_result=classification,
                interpretation_result=interpretation,
                latency_ms=latency_ms,
            )

            return self._build_response(
                session_id, classification, interpretation, audio_features, None, latency_ms
            )

        except Exception as exc:
            logger.error("analyze_audio failed for session %s: %s", session_id, exc)
            self._prom_errors_total += 1
            return {"session_id": session_id, "error": str(exc), "status": "error"}

    def analyze_text(
        self, text: str, session_id: Optional[str] = None
    ) -> Dict:
        """Text-only decode: emotion + Gottman + attachment + love language."""
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        self._gate_failures = []

        try:
            self._prom_sessions_total += 1
            from agent.modules.audio_analyzer import ProsodicFeatures
            audio_features = ProsodicFeatures()  # empty — text-only mode
            classifier = self._get_classifier()
            interpreter = self._get_interpreter()
            memory = self._get_memory()

            classification = classifier.classify(audio_features, text, None)

            # Gate 2: Emotion confidence check
            if classification.emotion_scores:
                classification.emotion_scores = self._check_emotion_confidence_gate(
                    classification.emotion_scores
                )

            # Gate 3: Classification confidence check
            gate3_result = self._check_classification_confidence_gate(classification)
            classification = gate3_result["classification"]

            # Gate 4: Safety gate
            self._check_safety_gate(classification)

            interpretation = interpreter.interpret(classification, audio_features, text, None)

            latency_ms = (time.time() - start_time) * 1000
            self._prom_latency_seconds_total += latency_ms / 1000.0
            memory.save_session(
                session_id=session_id,
                text_input=text,
                classification_result=classification,
                interpretation_result=interpretation,
                latency_ms=latency_ms,
            )

            return self._build_response(
                session_id, classification, interpretation, audio_features, None, latency_ms
            )

        except Exception as exc:
            logger.error("analyze_text failed: %s", exc)
            self._prom_errors_total += 1
            return {"session_id": session_id, "error": str(exc), "status": "error"}

    def analyze_session(
        self,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
        video_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Full multimodal session: audio + text + optional video."""
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        self._gate_failures = []

        try:
            self._prom_sessions_total += 1
            audio_analyzer = self._get_audio_analyzer()
            visual_analyzer = self._get_visual_analyzer()
            classifier = self._get_classifier()
            interpreter = self._get_interpreter()
            memory = self._get_memory()

            # Audio features
            from agent.modules.audio_analyzer import ProsodicFeatures
            audio_features = ProsodicFeatures()
            if audio_path:
                if audio_analyzer.validate_audio(audio_path):
                    audio_features = audio_analyzer.analyze_file(audio_path)
                    if not text:
                        try:
                            hf = self._get_hf_manager()
                            result = hf.transcribe(audio_path)
                            text = result.get("text", "") if isinstance(result, dict) else str(result)
                        except Exception as exc:
                            logger.warning("Transcription failed: %s", exc)
                else:
                    self._log_gate_error("audio_quality", f"Audio failed validation: {audio_path}")

            # Visual features
            visual_features = None
            if video_path and visual_analyzer.is_available():
                frame_features = visual_analyzer.analyze_video_file(video_path)
                visual_features = visual_analyzer.get_aggregate_features(frame_features)

            # Classify
            classification = classifier.classify(audio_features, text or "", visual_features)

            # Gate 2: Emotion confidence check
            if classification.emotion_scores:
                classification.emotion_scores = self._check_emotion_confidence_gate(
                    classification.emotion_scores
                )

            # Gate 3: Classification confidence check
            gate3_result = self._check_classification_confidence_gate(classification)
            classification = gate3_result["classification"]

            # Gate 4: Safety gate
            self._check_safety_gate(classification)

            # Interpret
            interpretation = interpreter.interpret(
                classification, audio_features, text or "", visual_features
            )

            latency_ms = (time.time() - start_time) * 1000
            self._prom_latency_seconds_total += latency_ms / 1000.0
            memory.save_session(
                session_id=session_id,
                audio_path=audio_path,
                text_input=text,
                has_video=video_path is not None,
                classification_result=classification,
                interpretation_result=interpretation,
                latency_ms=latency_ms,
            )

            return self._build_response(
                session_id, classification, interpretation, audio_features, visual_features, latency_ms
            )

        except Exception as exc:
            logger.error("analyze_session failed: %s", exc)
            self._prom_errors_total += 1
            return {"session_id": session_id, "error": str(exc), "status": "error"}

    def analyze_microphone(self, duration_seconds: int = 30) -> Dict:
        """Capture microphone audio and run full pipeline."""
        session_id = str(uuid.uuid4())
        start_time = time.time()
        self._gate_failures = []

        try:
            self._prom_sessions_total += 1
            audio_analyzer = self._get_audio_analyzer()
            classifier = self._get_classifier()
            interpreter = self._get_interpreter()
            memory = self._get_memory()

            audio_features = audio_analyzer.analyze_microphone(duration=duration_seconds)
            if not audio_features.audio_valid:
                return {
                    "session_id": session_id,
                    "error": audio_features.validation_message,
                    "status": "audio_invalid",
                }

            # Classify (text from microphone is empty unless Whisper processes live stream)
            classification = classifier.classify(audio_features, "", None)

            # Gate 2: Emotion confidence check
            if classification.emotion_scores:
                classification.emotion_scores = self._check_emotion_confidence_gate(
                    classification.emotion_scores
                )

            # Gate 3: Classification confidence check
            gate3_result = self._check_classification_confidence_gate(classification)
            classification = gate3_result["classification"]

            # Gate 4: Safety gate
            self._check_safety_gate(classification)

            interpretation = interpreter.interpret(classification, audio_features, "", None)

            latency_ms = (time.time() - start_time) * 1000
            self._prom_latency_seconds_total += latency_ms / 1000.0
            memory.save_session(
                session_id=session_id,
                classification_result=classification,
                interpretation_result=interpretation,
                latency_ms=latency_ms,
            )

            return self._build_response(
                session_id, classification, interpretation, audio_features, None, latency_ms
            )

        except Exception as exc:
            logger.error("analyze_microphone failed: %s", exc)
            self._prom_errors_total += 1
            return {"session_id": session_id, "error": str(exc), "status": "error"}

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return overall session statistics."""
        return self._get_memory().get_stats()

    def get_cost_report(self) -> Dict:
        """Return LLM cost breakdown by provider/model."""
        return self._get_memory().get_cost_report()

    def update_knowledge(self) -> Dict:
        """Trigger knowledge updater manually."""
        try:
            ku = self._get_knowledge_updater()
            result = ku.run()
            return {"status": "success", "new_papers": result}
        except Exception as exc:
            logger.error("Knowledge update failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def start_scheduler(self) -> None:
        """Start APScheduler for weekly knowledge updates."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                func=self.update_knowledge,
                trigger=CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="weekly_knowledge_update",
                name="Weekly knowledge base update",
                replace_existing=True,
            )
            self._scheduler.start()
            logger.info("APScheduler started — knowledge update scheduled weekly Sunday 02:00")
        except ImportError:
            logger.warning("APScheduler not installed — background scheduling disabled")
        except Exception as exc:
            logger.error("Failed to start scheduler: %s", exc)

    def get_prometheus_metrics(self) -> str:
        """Return Prometheus-format metrics string."""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest()
        except ImportError:
            # Return minimal text format with internal counters
            stats = self.get_stats()
            return (
                f"# HELP partner_decode_sessions_total Total sessions analyzed\n"
                f"# TYPE partner_decode_sessions_total counter\n"
                f"partner_decode_sessions_total {self._prom_sessions_total}\n"
                f"# HELP partner_decode_errors_total Total errors encountered\n"
                f"# TYPE partner_decode_errors_total counter\n"
                f"partner_decode_errors_total {self._prom_errors_total}\n"
                f"# HELP partner_decode_cost_usd_total Total LLM cost in USD\n"
                f"# TYPE partner_decode_cost_usd_total counter\n"
                f"partner_decode_cost_usd_total {self._prom_cost_usd_total:.6f}\n"
                f"# HELP partner_decode_latency_seconds_total Total processing time in seconds\n"
                f"# TYPE partner_decode_latency_seconds_total counter\n"
                f"partner_decode_latency_seconds_total {self._prom_latency_seconds_total:.2f}\n"
                f"# HELP partner_decode_avg_distress Average distress score\n"
                f"# TYPE partner_decode_avg_distress gauge\n"
                f"partner_decode_avg_distress {stats.get('avg_distress_score', 0.0)}\n"
            )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_response(
        self,
        session_id: str,
        classification,
        interpretation,
        audio_features,
        visual_features,
        latency_ms: float,
    ) -> Dict:
        """Build the standard JSON response dict."""
        try:
            interpreter = self._get_interpreter()
            formatted = interpreter.format_result(interpretation)
        except Exception:
            formatted = ""

        return {
            "session_id": session_id,
            "status": "success",
            "latency_ms": round(latency_ms, 2),
            "report": {
                "actual_feeling": interpretation.actual_feeling,
                "stated_vs_actual_gap": interpretation.stated_vs_actual_gap,
                "gottman_pattern_active": interpretation.gottman_pattern_active,
                "attachment_context": interpretation.attachment_context,
                "love_language_expressed": interpretation.love_language_expressed,
                "empathy_responses": interpretation.empathy_responses,
                "urgency_level": interpretation.urgency_level,
                "counseling_recommended": interpretation.counseling_recommended,
                "counseling_message": interpretation.counseling_message,
                "confidence_explanation": interpretation.confidence_explanation,
                "supporting_research": interpretation.supporting_research,
            },
            "classification": {
                "gottman_horsemen": classification.gottman_horsemen,
                "attachment_pattern": classification.attachment_pattern,
                "attachment_confidence": classification.attachment_confidence,
                "love_language_signals": classification.love_language_signals,
                "non_literal_detected": classification.non_literal_detected,
                "non_literal_phrase": classification.non_literal_phrase,
                "actual_emotional_state": classification.actual_emotional_state,
                "communication_style": classification.communication_style,
                "overall_distress_score": classification.overall_distress_score,
                "needs_counseling_flag": classification.needs_counseling_flag,
                "emotion_scores": classification.emotion_scores,
                "confidence": classification.confidence,
            },
            "audio_features": {
                "f0_mean": audio_features.f0_mean,
                "f0_std": audio_features.f0_std,
                "f0_range": audio_features.f0_range,
                "speech_rate": audio_features.speech_rate,
                "pause_count": audio_features.pause_count,
                "rms": audio_features.rms,
                "energy_mean": audio_features.energy_mean,
                "vocal_cues": audio_features.vocal_cues,
                "duration_seconds": audio_features.duration_seconds,
            },
            "visual_available": (
                visual_features is not None and visual_features.available
            ) if visual_features else False,
            "quality_gates": {
                "gate_failures": list(self._gate_failures),
                "classification_confidence": classification.confidence,
                "distress_score": classification.overall_distress_score,
                "counseling_flagged": classification.needs_counseling_flag,
            },
            "formatted_report": formatted,
            "meta": {
                "llm_provider": interpretation.llm_provider_used,
                "fallback_used": interpretation.fallback_used,
            },
        }
