# test-scenarios.md — partner-decode-female-agent

## Overview

8 end-to-end test scenarios covering the core agent behaviors:
non-literal expression detection, Gottman pattern detection, attachment pattern detection,
stonewalling, positive interaction, knowledge deduplication, offline fallback, and REST API.

---

## Scenario 1: Non-Literal Expression — "I'm Fine" Golden Path

**Scenario ID:** S01-NON-LITERAL-GOLDEN-PATH
**Type:** End-to-end audio analysis
**Priority:** Critical

**Setup:**
- Synthesize or record a 15-second audio clip of the phrase "I'm fine, don't worry"
  spoken with audible tension: fast speech rate (>4 syl/sec), elevated F0 variance (>35Hz std),
  slightly higher than baseline energy, no significant pauses
- Audio file: `tests/fixtures/im_fine_stressed.wav`

**Input:**
```
AudioAnalyzer.analyze_file("tests/fixtures/im_fine_stressed.wav")
BehaviorClassifier.classify(audio_features, "I'm fine, don't worry", None)
InterpretationEngine.interpret(classification, audio_features, "I'm fine, don't worry", None)
```

**Expected Output:**
1. `audio_features.f0_std > 25.0` (elevated pitch variance)
2. `audio_features.vocal_cues["has_emotional_stress"] == True`
3. `classification.non_literal_detected == True`
4. `classification.non_literal_phrase` contains "i'm fine"
5. `classification.actual_emotional_state == "stressed"`
6. `interpretation.actual_feeling` contains language about tension/stress/not-fine
7. `interpretation.urgency_level` in ["medium", "high"]
8. All 3 `interpretation.empathy_responses` are non-empty and phrased as options

**Failure Modes to Test:**
- Silent audio → `AudioAnalyzer.validate_audio()` returns False
- Very short audio (0.3s) → validation gate rejects

---

## Scenario 2: Gottman Contempt Detection

**Scenario ID:** S02-GOTTMAN-CONTEMPT
**Type:** Multimodal (text + visual cues)
**Priority:** Critical

**Setup:**
- Text: "Of course YOU would think that. That's just what you do. Wow."
- Audio features: monotone delivery, low F0 variance, clipped pauses
- Visual: expression_label="contempt", AU4=0.6, microexpression_detected=True

**Input:**
```python
text = "Of course YOU would think that. That's just what you do. Wow."
# Mock AudioAnalyzer returning low F0 variance, dismissive prosody
# Mock VisualAnalyzer returning contempt expression + AU4
classification = classifier.classify(audio_features, text, visual_features)
```

**Expected Output:**
1. `classification.gottman_horsemen["contempt"] > 0.5`
2. `classification.gottman_horsemen["criticism"] > 0.3` (overlapping pattern)
3. `classification.overall_distress_score > 0.6`
4. `interpretation.gottman_pattern_active` mentions "contempt"
5. `interpretation.urgency_level` == "high" or "counseling_needed"
6. If contempt score > 0.7: `interpretation.counseling_recommended == True`
7. `interpretation.counseling_message` is non-empty and compassionate

**Research Citation:**
Gottman (1994): Contempt is the single strongest predictor of divorce.

---

## Scenario 3: Anxious Attachment Pattern Detection

**Scenario ID:** S03-ANXIOUS-ATTACHMENT
**Type:** Audio + text analysis
**Priority:** High

**Setup:**
- Audio: high F0 variance (std > 45Hz), fast speech rate (>4 syl/sec), multiple rising
  intonation patterns at statement ends
- Text: "Are you sure you still want to be with me? I just feel like you've been
  distant lately. Do you still care? Please tell me everything is okay."

**Input:**
```python
audio = ProsodicFeatures(f0_std=48.0, speech_rate=4.2, vocal_cues={"has_voice_rise": True}, ...)
text = "Are you sure you still want to be with me? ..."
classification = classifier.classify(audio, text, None)
```

**Expected Output:**
1. `classification.attachment_pattern == "anxious"`
2. `classification.attachment_confidence > 0.5`
3. `classification.love_language_signals["words_of_affirmation"] > 0.3`
4. `interpretation.attachment_context` references anxious attachment / reassurance-seeking
5. `interpretation.love_language_expressed` contains "words of affirmation"
6. Empathy responses include reassurance-giving options (not commands)
7. `interpretation.urgency_level` in ["medium", "high"]

---

## Scenario 4: Stonewalling / Silent Treatment Detection

**Scenario ID:** S04-STONEWALLING
**Type:** Audio + minimal text
**Priority:** High

**Setup:**
- Audio: very low RMS energy (< 0.003), very low F0 variance, minimal speech
- Text: "ok" (single word response)
- Visual (optional): eyes averted, minimal facial movement

**Input:**
```python
audio = ProsodicFeatures(rms=0.002, f0_std=4.0, energy_mean=0.001, ...)
text = "ok"
classification = classifier.classify(audio, text, None)
```

**Expected Output:**
1. `classification.gottman_horsemen["stonewalling"] > 0.5`
2. `classification.communication_style == "withdrawn"`
3. `classification.overall_distress_score > 0.6`
4. `interpretation.gottman_pattern_active` mentions "stonewalling"
5. `interpretation.empathy_responses` include offers to give space / non-demanding presence
6. `classification.needs_counseling_flag == True` (distress > 0.8 threshold)

**Notes:**
Stonewalling research: Levenson & Gottman (1994) linked stonewalling to physiological flooding (HR > 100 BPM).

---

## Scenario 5: Positive Secure Interaction

**Scenario ID:** S05-POSITIVE-SECURE
**Type:** Audio + text
**Priority:** Medium

**Setup:**
- Audio: calm delivery, moderate F0 variance (10-25Hz), measured speech rate (2.5-3.5 syl/sec)
- Text: "I understand why you feel that way, and I really appreciate you telling me.
  Thank you for being honest. I love you and we can figure this out together."

**Input:**
```python
audio = ProsodicFeatures(f0_std=18.0, speech_rate=3.0, rms=0.015, ...)
text = "I understand why you feel that way..."
classification = classifier.classify(audio, text, None)
```

**Expected Output:**
1. All `classification.gottman_horsemen` scores < 0.2
2. `classification.attachment_pattern == "secure"`
3. `classification.love_language_signals["words_of_affirmation"] > 0.3`
4. `classification.overall_distress_score < 0.3`
5. `classification.needs_counseling_flag == False`
6. `interpretation.urgency_level == "low"`
7. `interpretation.counseling_recommended == False`

---

## Scenario 6: Knowledge Crawler Deduplication

**Scenario ID:** S06-KNOWLEDGE-DEDUP
**Type:** Knowledge updater unit test
**Priority:** High

**Setup:**
- Run `KnowledgeUpdater.run()` with mocked ArXiv + S2 responses returning 5 papers
- Store all 5 SHA256 hashes in the database (simulate first run)
- Run `KnowledgeUpdater.run()` again with the same mocked papers

**Input:**
```python
updater = KnowledgeUpdater(memory_manager=mock_memory)
# First run
n1 = updater.run()  # Should add 5 papers
# Second run with identical response
n2 = updater.run()  # Should add 0 papers (dedup)
```

**Expected Output:**
1. First run: `n1 == 5`
2. Second run: `n2 == 0`
3. `mock_memory.add_knowledge_hash` called exactly 5 times total
4. `mock_memory.has_knowledge_hash` returns True for all 5 URLs on second run
5. SECOND-KNOWLEDGE-BRAIN.md contains new rows only from first run
6. Knowledge Update Log shows 2 entries: one with 5, one with 0

---

## Scenario 7: All LLM Providers Down — Offline Fallback

**Scenario ID:** S07-LLM-FALLBACK
**Type:** Resilience test
**Priority:** Critical

**Setup:**
- Mock `LLMClient.complete()` to raise `RuntimeError("All providers failed")`
- Run full analysis pipeline with valid audio and text

**Input:**
```python
with patch("tools.llm_client.LLMClient.complete", side_effect=RuntimeError("All providers failed")):
    result = orchestrator.analyze_text("I'm fine, everything is great.", session_id="test-fallback")
```

**Expected Output:**
1. No exception raised — agent returns gracefully
2. `result["status"] == "success"`
3. `result["meta"]["fallback_used"] == True`
4. `result["report"]["actual_feeling"]` is non-empty (from FALLBACK_INTERPRETATIONS)
5. `result["report"]["empathy_responses"]` has exactly 3 items
6. `result["report"]["confidence_explanation"]` mentions "offline mode" or "fallback"
7. Session is still saved to database

---

## Scenario 8: REST API Full Integration

**Scenario ID:** S08-REST-API-FULL
**Type:** Integration test
**Priority:** Critical

**Setup:**
- Start FastAPI test client: `TestClient(app)`
- Prepare a valid audio file (WAV format, 10 seconds, human voice)
- Text: "I don't know, I guess it's fine."

**Request:**
```http
POST /api/v1/analyze/session
Content-Type: multipart/form-data

audio_file=@tests/fixtures/sample_10s.wav
text=I don't know, I guess it's fine.
session_id=test-integration-001
```

**Expected Response (HTTP 200):**
```json
{
  "session_id": "test-integration-001",
  "status": "success",
  "latency_ms": <number>,
  "report": {
    "actual_feeling": "<non-empty string>",
    "empathy_responses": ["<string>", "<string>", "<string>"],
    "urgency_level": "low|medium|high|counseling_needed",
    "counseling_recommended": <boolean>,
    "supporting_research": ["<citation>"]
  },
  "classification": {
    "gottman_horsemen": {
      "criticism": <0.0-1.0>,
      "contempt": <0.0-1.0>,
      "defensiveness": <0.0-1.0>,
      "stonewalling": <0.0-1.0>
    },
    "attachment_pattern": "secure|anxious|avoidant",
    "non_literal_detected": <boolean>,
    "overall_distress_score": <0.0-1.0>
  }
}
```

**Assertions:**
1. HTTP status code == 200
2. Response JSON parses without error
3. All required fields present (report, classification, audio_features)
4. `latency_ms > 0`
5. `empathy_responses` length == 3
6. `overall_distress_score` between 0.0 and 1.0
7. Session saved to database: `GET /api/v1/sessions` returns session_id in list
8. `GET /api/v1/stats` returns `total_sessions >= 1`
