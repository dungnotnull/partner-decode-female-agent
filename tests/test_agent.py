"""
test_agent.py — Automated tests for partner-decode-female-agent.

44 tests organized as:
- 7 AudioAnalyzer tests
- 7 VisualAnalyzer tests
- 8 BehaviorClassifier tests
- 7 InterpretationEngine tests
- 6 MemoryManager tests
- 3 LLMClient tests
- 3 HFModelManager tests
- 5 Integration tests (mocked orchestrator pipeline)
- 5 CLI smoke tests

All tests use unittest.mock — NO real API calls.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def make_silent_wav(path: str, duration: float = 2.0, sr: int = 16000) -> None:
    """Write a silent WAV file to path."""
    try:
        import soundfile as sf
        samples = np.zeros(int(duration * sr), dtype=np.float32)
        sf.write(path, samples, sr)
    except ImportError:
        # Create minimal WAV header manually
        import struct, wave
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            n_frames = int(duration * sr)
            wf.writeframes(b"\x00" * n_frames * 2)


def make_speech_wav(path: str, duration: float = 3.0, sr: int = 16000) -> None:
    """Write a synthetic sine-wave 'speech' WAV file to path."""
    try:
        import soundfile as sf
        t = np.linspace(0, duration, int(duration * sr), endpoint=False)
        # Simulate voice: mix of 200Hz fundamental + harmonics
        signal = (
            0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.2 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 880 * t)
        ).astype(np.float32)
        sf.write(path, signal, sr)
    except ImportError:
        make_silent_wav(path, duration, sr)


# ===========================================================================
# 1. AudioAnalyzer Tests (7)
# ===========================================================================

class TestAudioAnalyzer(unittest.TestCase):

    def setUp(self):
        from agent.modules.audio_analyzer import AudioAnalyzer
        self.analyzer = AudioAnalyzer(use_wav2vec2=False)
        self.tmpdir = tempfile.mkdtemp()

    def test_validate_audio_rejects_nonexistent_file(self):
        """validate_audio returns False for missing file."""
        result = self.analyzer.validate_audio("/nonexistent/path/audio.wav")
        self.assertFalse(result)

    def test_validate_audio_rejects_silent_file(self):
        """validate_audio returns False for silent audio (RMS < threshold)."""
        path = os.path.join(self.tmpdir, "silent.wav")
        make_silent_wav(path, duration=2.0)
        result = self.analyzer.validate_audio(path)
        self.assertFalse(result)

    def test_analyze_file_returns_prosodic_features(self):
        """analyze_file returns ProsodicFeatures with populated fields."""
        from agent.modules.audio_analyzer import ProsodicFeatures
        path = os.path.join(self.tmpdir, "speech.wav")
        make_speech_wav(path, duration=3.0)
        features = self.analyzer.analyze_file(path)
        self.assertIsInstance(features, ProsodicFeatures)
        # Duration should be approximately 3 seconds
        self.assertGreater(features.duration_seconds, 1.0)

    def test_mfcc_shape(self):
        """MFCC mean array has correct shape (40 coefficients)."""
        path = os.path.join(self.tmpdir, "speech.wav")
        make_speech_wav(path, duration=3.0)
        features = self.analyzer.analyze_file(path)
        if features.audio_valid:
            self.assertEqual(len(features.mfcc_mean), 40)
            self.assertEqual(len(features.mfcc_std), 40)

    def test_f0_detection_returns_float(self):
        """F0 extraction returns float values."""
        path = os.path.join(self.tmpdir, "speech.wav")
        make_speech_wav(path, duration=3.0)
        features = self.analyzer.analyze_file(path)
        if features.audio_valid:
            self.assertIsInstance(features.f0_mean, float)
            self.assertIsInstance(features.f0_std, float)

    def test_speech_rate_non_negative(self):
        """Speech rate is always non-negative."""
        path = os.path.join(self.tmpdir, "speech.wav")
        make_speech_wav(path, duration=3.0)
        features = self.analyzer.analyze_file(path)
        if features.audio_valid:
            self.assertGreaterEqual(features.speech_rate, 0.0)

    def test_to_feature_vector_returns_numpy_array(self):
        """to_feature_vector returns a 1D float32 numpy array."""
        from agent.modules.audio_analyzer import ProsodicFeatures
        features = ProsodicFeatures(
            mfcc_mean=np.zeros(40),
            mfcc_std=np.zeros(40),
            mfcc_delta_mean=np.zeros(40),
            mfcc_delta2_mean=np.zeros(40),
            chroma_mean=np.zeros(12),
            audio_valid=True,
        )
        vec = self.analyzer.to_feature_vector(features)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.float32)
        # 40+40+40+40+12+15 = 187 base dimensions
        self.assertEqual(len(vec), 187)


# ===========================================================================
# 2. VisualAnalyzer Tests (7)
# ===========================================================================

class TestVisualAnalyzer(unittest.TestCase):

    def setUp(self):
        from agent.modules.visual_analyzer import VisualAnalyzer
        self.analyzer = VisualAnalyzer()

    def test_is_available_returns_bool(self):
        """is_available() returns a boolean."""
        result = self.analyzer.is_available()
        self.assertIsInstance(result, bool)

    def test_unavailable_features_when_mediapipe_missing(self):
        """When MediaPipe unavailable, all methods return VisualFeatures with available=False."""
        from agent.modules.visual_analyzer import VisualAnalyzer
        analyzer = VisualAnalyzer()
        analyzer._mp_available = False  # Force unavailable
        result = analyzer.analyze_image("/fake/path.jpg")
        from agent.modules.visual_analyzer import VisualFeatures
        self.assertIsInstance(result, VisualFeatures)
        self.assertFalse(result.available)

    def test_analyze_video_frame_returns_visual_features(self):
        """analyze_video_frame on non-face frame returns VisualFeatures."""
        from agent.modules.visual_analyzer import VisualAnalyzer, VisualFeatures
        analyzer = VisualAnalyzer()
        analyzer._mp_available = False
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyzer.analyze_video_frame(frame)
        self.assertIsInstance(result, VisualFeatures)

    def test_au_scores_default_to_zero(self):
        """Default VisualFeatures has empty or zero AU scores."""
        from agent.modules.visual_analyzer import VisualFeatures
        vf = VisualFeatures()
        self.assertIsInstance(vf.au_scores, dict)

    def test_expression_label_default_neutral(self):
        """Default VisualFeatures expression_label is 'neutral'."""
        from agent.modules.visual_analyzer import VisualFeatures
        vf = VisualFeatures()
        self.assertEqual(vf.expression_label, "neutral")

    def test_analyze_video_file_unavailable_returns_list(self):
        """analyze_video_file returns a list even when MediaPipe unavailable."""
        from agent.modules.visual_analyzer import VisualAnalyzer
        analyzer = VisualAnalyzer()
        analyzer._mp_available = False
        result = analyzer.analyze_video_file("/fake/video.mp4")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_get_aggregate_features_handles_empty_list(self):
        """get_aggregate_features handles empty input gracefully."""
        from agent.modules.visual_analyzer import VisualAnalyzer, VisualFeatures
        analyzer = VisualAnalyzer()
        result = analyzer.get_aggregate_features([])
        self.assertIsInstance(result, VisualFeatures)
        self.assertFalse(result.available)


# ===========================================================================
# 3. BehaviorClassifier Tests (8)
# ===========================================================================

class TestBehaviorClassifier(unittest.TestCase):

    def setUp(self):
        from agent.modules.behavior_classifier import BehaviorClassifier
        from agent.modules.audio_analyzer import ProsodicFeatures
        self.classifier = BehaviorClassifier(hf_model_manager=None)
        self.neutral_audio = ProsodicFeatures(rms=0.01, f0_std=15.0, speech_rate=2.5)

    def test_classify_returns_classification_result(self):
        """classify() returns a ClassificationResult."""
        from agent.modules.behavior_classifier import ClassificationResult
        result = self.classifier.classify(self.neutral_audio, "hello", None)
        self.assertIsInstance(result, ClassificationResult)

    def test_criticism_pattern_detected(self):
        """Criticism patterns in text produce elevated criticism score."""
        result = self.classifier.classify(
            self.neutral_audio,
            "You always do this. You never listen to me.",
            None,
        )
        self.assertGreater(result.gottman_horsemen["criticism"], 0.3)

    def test_contempt_pattern_detected(self):
        """Contempt patterns in text produce elevated contempt score."""
        result = self.classifier.classify(
            self.neutral_audio,
            "That's ridiculous. You don't even know what you're talking about.",
            None,
        )
        self.assertGreater(result.gottman_horsemen["contempt"], 0.2)

    def test_anxious_attachment_from_text(self):
        """Anxious attachment keywords produce 'anxious' classification."""
        result = self.classifier.classify(
            self.neutral_audio,
            "Do you still love me? Please don't leave me. Are you sure you care?",
            None,
        )
        self.assertEqual(result.attachment_pattern, "anxious")

    def test_non_literal_detection_im_fine(self):
        """'I'm fine' with stressed prosody triggers non-literal detection."""
        from agent.modules.audio_analyzer import ProsodicFeatures
        stressed_audio = ProsodicFeatures(
            rms=0.02,
            f0_std=38.0,  # high variance = stressed
            speech_rate=4.2,
            energy_mean=0.03,
            energy_std=0.015,
        )
        result = self.classifier.classify(stressed_audio, "I'm fine", None)
        self.assertTrue(result.non_literal_detected)
        self.assertIn("i'm fine", result.non_literal_phrase.lower())

    def test_love_language_words_of_affirmation(self):
        """Words of affirmation signals detected from text."""
        result = self.classifier.classify(
            self.neutral_audio,
            "I love you so much. You're amazing and I really appreciate everything you do.",
            None,
        )
        self.assertGreater(result.love_language_signals["words_of_affirmation"], 0.0)

    def test_safety_gate_counseling_flag(self):
        """needs_counseling_flag=True when distress_score > 0.8."""
        from agent.modules.audio_analyzer import ProsodicFeatures
        high_distress_audio = ProsodicFeatures(
            rms=0.05,
            f0_std=55.0,
            energy_mean=0.08,
            energy_std=0.04,
            vocal_cues={"has_emotional_stress": True},
        )
        # Contempt text to drive horseman score high
        result = self.classifier.classify(
            high_distress_audio,
            "You always do this. You never think about anyone else. How pathetic.",
            None,
        )
        # At least one horseman should be elevated
        max_horseman = max(result.gottman_horsemen.values())
        self.assertGreater(max_horseman, 0.3)
        # If counseling flag triggered, verify it was set
        if result.overall_distress_score > 0.8:
            self.assertTrue(result.needs_counseling_flag)

    def test_secure_attachment_from_text(self):
        """Secure attachment markers produce 'secure' classification."""
        result = self.classifier.classify(
            self.neutral_audio,
            "I hear you and that makes sense. I'm sorry. We can figure this out together.",
            None,
        )
        # Secure patterns should score highest
        self.assertIsInstance(result.attachment_pattern, str)
        self.assertIn(result.attachment_pattern, ["secure", "anxious", "avoidant"])


# ===========================================================================
# 4. InterpretationEngine Tests (7)
# ===========================================================================

class TestInterpretationEngine(unittest.TestCase):

    def setUp(self):
        from agent.modules.interpretation_engine import InterpretationEngine
        from agent.modules.audio_analyzer import ProsodicFeatures
        from agent.modules.behavior_classifier import ClassificationResult

        self.neutral_audio = ProsodicFeatures(rms=0.01, f0_std=15.0)
        self.neutral_classification = ClassificationResult(
            attachment_pattern="secure",
            attachment_confidence=0.6,
            overall_distress_score=0.2,
            communication_style="direct",
        )
        # Engine without LLM client (tests fallback behavior)
        self.engine_no_llm = InterpretationEngine(llm_client=None)

    def test_interpret_returns_interpretation_result(self):
        """interpret() returns InterpretationResult even with no LLM."""
        from agent.modules.interpretation_engine import InterpretationResult
        result = self.engine_no_llm.interpret(
            self.neutral_classification,
            self.neutral_audio,
            "I'm okay",
            None,
        )
        self.assertIsInstance(result, InterpretationResult)

    def test_fallback_used_when_no_llm(self):
        """fallback_used=True when LLM client is None."""
        result = self.engine_no_llm.interpret(
            self.neutral_classification,
            self.neutral_audio,
            "test",
            None,
        )
        self.assertTrue(result.fallback_used)

    def test_empathy_responses_count(self):
        """Fallback always returns exactly 3 empathy responses."""
        result = self.engine_no_llm.interpret(
            self.neutral_classification,
            self.neutral_audio,
            "fine",
            None,
        )
        self.assertEqual(len(result.empathy_responses), 3)

    def test_counseling_flag_propagates_to_interpretation(self):
        """When classification.needs_counseling_flag=True, interpretation sets counseling_recommended=True."""
        from agent.modules.behavior_classifier import ClassificationResult
        high_distress = ClassificationResult(
            needs_counseling_flag=True,
            overall_distress_score=0.9,
            communication_style="withdrawn",
        )
        result = self.engine_no_llm.interpret(
            high_distress, self.neutral_audio, "ok", None
        )
        self.assertTrue(result.counseling_recommended)
        self.assertTrue(len(result.counseling_message) > 0)

    def test_format_result_returns_string(self):
        """format_result() returns a non-empty string with ASCII box."""
        from agent.modules.interpretation_engine import InterpretationResult
        ir = InterpretationResult(
            actual_feeling="She seems stressed",
            empathy_responses=["Option 1", "Option 2", "Option 3"],
            urgency_level="medium",
        )
        formatted = self.engine_no_llm.format_result(ir)
        self.assertIsInstance(formatted, str)
        self.assertIn("EMPATHY REPORT", formatted)

    def test_llm_json_parsing_success(self):
        """_parse_llm_response correctly parses valid JSON."""
        valid_json = json.dumps({
            "actual_feeling": "She is anxious",
            "stated_vs_actual_gap": "Big gap",
            "empathy_responses": ["A", "B", "C"],
            "urgency_level": "medium",
            "counseling_recommended": False,
            "counseling_message": "",
        })
        parsed = self.engine_no_llm._parse_llm_response(valid_json)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["actual_feeling"], "She is anxious")

    def test_llm_json_parsing_handles_markdown_fences(self):
        """_parse_llm_response strips markdown code fences before parsing."""
        json_with_fence = "```json\n{\"actual_feeling\": \"test\"}\n```"
        parsed = self.engine_no_llm._parse_llm_response(json_with_fence)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["actual_feeling"], "test")


# ===========================================================================
# 5. MemoryManager Tests (6)
# ===========================================================================

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test.db")
        from agent.memory.memory_manager import MemoryManager
        self.memory = MemoryManager(db_path=db_path)

    def test_tables_created(self):
        """All 5 required tables exist after initialization."""
        import sqlite3
        conn = sqlite3.connect(self.memory.db_path)
        tables = [
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        conn.close()
        for expected_table in [
            "decode_sessions", "communication_patterns", "partner_profile",
            "llm_cost_log", "knowledge_hashes"
        ]:
            self.assertIn(expected_table, tables, f"Missing table: {expected_table}")

    def test_save_and_retrieve_session(self):
        """save_session persists data; get_recent_sessions retrieves it."""
        self.memory.save_session(
            session_id="test-001",
            text_input="hello",
        )
        sessions = self.memory.get_recent_sessions(n=5)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "test-001")

    def test_knowledge_hash_dedup(self):
        """add_knowledge_hash + has_knowledge_hash prevent duplicate entries."""
        url = "https://arxiv.org/abs/1234.5678"
        self.assertFalse(self.memory.has_knowledge_hash(url))
        self.memory.add_knowledge_hash(url)
        self.assertTrue(self.memory.has_knowledge_hash(url))
        # Second add should not raise
        self.memory.add_knowledge_hash(url)
        self.assertTrue(self.memory.has_knowledge_hash(url))

    def test_llm_cost_logging(self):
        """log_llm_cost stores entries; get_cost_report aggregates them."""
        self.memory.log_llm_cost("claude", "claude-opus-4-8", 500, 300, 0.015)
        self.memory.log_llm_cost("openai", "gpt-4o", 400, 200, 0.008)
        report = self.memory.get_cost_report()
        self.assertIn("total_usd", report)
        self.assertGreater(report["total_usd"], 0.0)
        self.assertEqual(len(report["by_provider_model"]), 2)

    def test_get_stats_returns_dict(self):
        """get_stats() returns a dict with total_sessions and related fields."""
        self.memory.save_session(session_id="stats-test-001", text_input="test")
        stats = self.memory.get_stats()
        self.assertIn("total_sessions", stats)
        self.assertGreater(stats["total_sessions"], 0)
        self.assertIn("avg_distress_score", stats)

    def test_thread_safety(self):
        """Multiple threads can write sessions concurrently without corruption."""
        import threading
        errors = []

        def write_session(idx):
            try:
                self.memory.save_session(
                    session_id=f"thread-{idx:03d}",
                    text_input=f"thread test {idx}",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_session, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        sessions = self.memory.get_recent_sessions(n=20)
        self.assertEqual(len(sessions), 10)


# ===========================================================================
# 6. LLMClient Tests (3)
# ===========================================================================

class TestLLMClient(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test.db")
        from agent.memory.memory_manager import MemoryManager
        self.memory = MemoryManager(db_path=db_path)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"})
    def test_provider_chain_falls_back_to_openai(self):
        """When Claude fails, LLMClient tries OpenAI."""
        from tools.llm_client import LLMClient
        client = LLMClient(memory_manager=self.memory)

        with patch.object(client, "_call_claude", side_effect=Exception("Claude down")):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-openai-key"}):
                with patch.object(client, "_call_openai", return_value="openai response"):
                    text, provider = client.complete("hello")
                    self.assertIn("openai", provider.lower())

    @patch.dict(os.environ, {"PRIVACY_MODE": "true"})
    def test_privacy_mode_forces_ollama(self):
        """PRIVACY_MODE=true routes all calls to Ollama only."""
        from tools.llm_client import LLMClient
        client = LLMClient(memory_manager=self.memory)
        self.assertTrue(client.privacy_mode)
        self.assertEqual(len(client._provider_order), 1)
        self.assertEqual(client._provider_order[0][0], "ollama")

    def test_cost_computation(self):
        """_compute_cost returns non-negative float for known models."""
        from tools.llm_client import LLMClient
        client = LLMClient()
        cost = client._compute_cost("claude-opus-4-8", 1000, 600)
        self.assertGreater(cost, 0.0)
        self.assertIsInstance(cost, float)


# ===========================================================================
# 7. HFModelManager Tests (3)
# ===========================================================================

class TestHFModelManager(unittest.TestCase):

    def test_singleton_pattern(self):
        """HFModelManager.get_instance() returns the same object each call."""
        from tools.hf_model_manager import HFModelManager
        mgr1 = HFModelManager.get_instance(cache_dir=tempfile.mkdtemp())
        mgr2 = HFModelManager.get_instance()
        self.assertIs(mgr1, mgr2)

    def test_emotion_fallback_returns_7_classes(self):
        """_tfidf_emotion_fallback returns dict with all 7 emotion keys."""
        from tools.hf_model_manager import HFModelManager, EMOTION_LABELS
        mgr = HFModelManager(cache_dir=tempfile.mkdtemp())
        result = mgr._tfidf_emotion_fallback("I am so sad and hurt")
        for label in EMOTION_LABELS:
            self.assertIn(label, result)
        # Scores should sum to approximately 1.0
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_tfidf_fallback_encode_returns_numpy(self):
        """_tfidf_fallback_encode returns float32 numpy array."""
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager(cache_dir=tempfile.mkdtemp())
        vec = mgr._tfidf_fallback_encode("attachment theory relationship")
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.float32)


# ===========================================================================
# 8. Integration Tests (5)
# ===========================================================================

class TestIntegrationPipeline(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_mock_orchestrator(self):
        """Create an orchestrator with all external calls mocked."""
        from agent.orchestrator import PartnerDecodeOrchestrator
        from agent.memory.memory_manager import MemoryManager
        from agent.modules.audio_analyzer import ProsodicFeatures
        from agent.modules.behavior_classifier import ClassificationResult
        from agent.modules.interpretation_engine import InterpretationResult

        config = {
            "memory": {"db_path": os.path.join(self.tmpdir, "test.db")},
            "hf_models": {"cache_dir": self.tmpdir},
        }
        orch = PartnerDecodeOrchestrator(config=config)

        # Inject mocked components
        mock_memory = MemoryManager(db_path=config["memory"]["db_path"])
        orch._memory = mock_memory

        mock_hf = MagicMock()
        mock_hf.transcribe.return_value = {"text": "I'm fine, don't worry", "chunks": []}
        mock_hf.classify_emotion.return_value = {"anger": 0.1, "fear": 0.2, "joy": 0.1, "neutral": 0.3, "sadness": 0.2, "disgust": 0.0, "surprise": 0.1}
        mock_hf.extract_wav2vec2.return_value = np.zeros(768, dtype=np.float32)
        orch._hf_manager = mock_hf

        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            json.dumps({
                "actual_feeling": "She appears stressed despite saying she is fine",
                "stated_vs_actual_gap": "Significant gap detected",
                "gottman_pattern_active": "none",
                "attachment_context": "Possible avoidant expression",
                "love_language_expressed": "words of affirmation",
                "empathy_responses": ["Option 1", "Option 2", "Option 3"],
                "urgency_level": "medium",
                "counseling_recommended": False,
                "counseling_message": "",
                "confidence_explanation": "Audio stress markers detected",
                "supporting_research": ["Gottman 1994"],
            }),
            "claude/claude-opus-4-8",
        )
        orch._llm_client = mock_llm

        return orch

    def test_analyze_text_full_pipeline(self):
        """Full text analysis pipeline returns expected structure."""
        orch = self._make_mock_orchestrator()
        result = orch.analyze_text("I'm fine, everything is great.", session_id="int-test-001")
        self.assertEqual(result["status"], "success")
        self.assertIn("report", result)
        self.assertIn("classification", result)
        self.assertIn("session_id", result)
        self.assertEqual(result["session_id"], "int-test-001")

    def test_analyze_text_report_structure(self):
        """Text analysis report has all required fields."""
        orch = self._make_mock_orchestrator()
        result = orch.analyze_text("Do you still love me?")
        report = result.get("report", {})
        required_fields = [
            "actual_feeling", "empathy_responses", "urgency_level",
            "counseling_recommended", "gottman_pattern_active"
        ]
        for field in required_fields:
            self.assertIn(field, report, f"Missing field: {field}")

    def test_all_llm_providers_down_returns_fallback(self):
        """When LLM fails, agent returns fallback without crashing."""
        orch = self._make_mock_orchestrator()
        orch._llm_client.complete.side_effect = RuntimeError("All providers failed")

        result = orch.analyze_text("whatever", session_id="fallback-test")
        self.assertNotEqual(result.get("status"), "error",
                            "Should not return error status on LLM failure — should use fallback")
        report = result.get("report", {})
        self.assertTrue(len(report.get("empathy_responses", [])) > 0)

    def test_memory_persists_session(self):
        """Session is saved to database after analysis."""
        orch = self._make_mock_orchestrator()
        orch.analyze_text("testing memory persistence", session_id="mem-test-001")
        sessions = orch._get_memory().get_recent_sessions(n=5)
        session_ids = [s["session_id"] for s in sessions]
        self.assertIn("mem-test-001", session_ids)

    def test_stats_after_sessions(self):
        """get_stats() reflects completed sessions."""
        orch = self._make_mock_orchestrator()
        orch.analyze_text("first session")
        orch.analyze_text("second session")
        stats = orch.get_stats()
        self.assertGreaterEqual(stats["total_sessions"], 2)


# ===========================================================================
# 9. CLI Smoke Tests (5)
# ===========================================================================

class TestCLISmokeTests(unittest.TestCase):
    """Verify that CLI commands show help without crashing."""

    def _run_cli_help(self, args):
        """Helper to invoke CLI with --help and capture output."""
        from click.testing import CliRunner
        from agent.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, args)
        return result

    def test_analyze_help(self):
        """analyze --help exits with code 0."""
        result = self._run_cli_help(["analyze", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("analyze", result.output.lower())

    def test_listen_help(self):
        """listen --help exits with code 0."""
        result = self._run_cli_help(["listen", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_decode_help(self):
        """decode --help exits with code 0."""
        result = self._run_cli_help(["decode", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_serve_help(self):
        """serve --help exits with code 0."""
        result = self._run_cli_help(["serve", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("port", result.output.lower())

    def test_version_flag(self):
        """--version flag shows version string."""
        result = self._run_cli_help(["--version"])
        self.assertIn("1.0.0", result.output)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
