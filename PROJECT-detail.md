# PROJECT-detail.md — partner-decode-female-agent

## Executive Summary

The partner-decode-female-agent is a multimodal AI application that decodes the emotional intent and communication patterns of a female partner through voice audio, text transcripts, and optional video analysis. The agent is grounded in three validated relationship-science frameworks: Gottman's Four Horsemen (predicting relationship distress), Ainsworth's Attachment Theory (classifying relational bonding patterns), and Chapman's Five Love Languages (identifying emotional expression preferences). The agent never tells users what to say; it only helps them understand. Every output is framed as an empathy and self-improvement tool. When high-conflict patterns are detected, the agent mandatorily recommends professional couples counseling.

## Problem Statement

Communication breakdown is the leading cause of relationship dissatisfaction. Research by Gottman (1994) shows four specific communication patterns — criticism, contempt, defensiveness, stonewalling — predict relationship dissolution with 93% accuracy up to 14 years in advance. The challenge for partners is that these patterns are often expressed indirectly, through tone, pacing, facial microexpressions, and non-literal language ("I'm fine" meaning "I'm not fine"). No existing consumer tool combines prosodic speech analysis, facial action unit detection, NLP emotion classification, and validated relationship-science frameworks into a unified, privacy-aware agent. This agent fills that gap, providing professional-grade insights previously available only in couples therapy.

## Target Users and Use Cases

**Primary Users:**
- Partners seeking to understand communication breakdowns in romantic relationships
- Therapists and counselors wanting a digital aid for session analysis (with consent)
- Relationship researchers studying multimodal emotional expression

**Use Cases:**

| Trigger | Agent Response |
|---------|---------------|
| User uploads 30-second audio of partner saying "I'm fine, don't worry" | Detects non-literal expression via prosodic stress markers; outputs empathy report explaining the gap |
| User types a text message received from partner | Emotion classification + hedging language detection + love language signal identification |
| User runs full multimodal session (audio + video + text) | Complete Four Horsemen + Attachment + Love Language report with urgency level |
| High-conflict session detected (distress > 0.8) | Automatic counseling recommendation with supportive framing |
| Weekly knowledge update | SECOND-KNOWLEDGE-BRAIN.md updated with latest relationship science papers |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    partner-decode-female-agent                          │
│                                                                         │
│  Input Layer                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ Audio    │  │  Text    │  │  Video   │  │   Microphone         │   │
│  │  File    │  │ Message  │  │  Frame   │  │   (live stream)      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘   │
│       │             │             │                    │               │
│  Feature Extraction Layer                                               │
│  ┌────▼─────────────┐  ┌──────────▼──────┐  ┌────────▼──────────────┐ │
│  │  AudioAnalyzer   │  │ VisualAnalyzer   │  │   HFModelManager      │ │
│  │ MFCC(40+d+d2)    │  │ MediaPipe 468    │  │ whisper → transcript  │ │
│  │ F0/speech_rate   │  │ AU1..AU26        │  │ emotion-distilroberta │ │
│  │ wav2vec2 768-dim │  │ microexpression  │  │ wav2vec2 embedding    │ │
│  └────┬─────────────┘  └──────────┬──────┘  └────────┬──────────────┘ │
│       │                           │                   │               │
│  ┌────▼───────────────────────────▼───────────────────▼─────────────┐  │
│  │                    BehaviorClassifier                             │  │
│  │  Four Horsemen (Gottman) │ Attachment Pattern (Ainsworth)        │  │
│  │  Love Language Signals   │ Non-Literal Expression Decoder        │  │
│  └────────────────────────────────┬──────────────────────────────────┘ │
│                                   │                                    │
│                          Safety Gate Check                             │
│                   (distress > 0.8 → counseling_flag)                  │
│                                   │                                    │
│  ┌────────────────────────────────▼──────────────────────────────────┐ │
│  │                  InterpretationEngine                             │ │
│  │   Claude API (primary) │ OpenAI (fallback) │ Ollama (offline)    │ │
│  │   EMPATHY_REPORT_PROMPT → structured JSON report                 │ │
│  └────────────────────────────────┬──────────────────────────────────┘ │
│                                   │                                    │
│  ┌────────────────────────────────▼──────────────────────────────────┐ │
│  │                    MemoryManager (SQLite WAL)                     │ │
│  │  decode_sessions │ communication_patterns │ partner_profile       │ │
│  │  llm_cost_log    │ knowledge_hashes                               │ │
│  └────────────────────────────────┬──────────────────────────────────┘ │
│                                   │                                    │
│  Output: Structured JSON + ASCII empathy report + urgency level        │
└─────────────────────────────────────────────────────────────────────────┘
         │              │              │
   LLM API          HuggingFace    External APIs
 (llm_client)    (hf_model_mgr)  (ArXiv/Semantic Scholar)
```

---

## Full Module Catalog

### Module 1: `agent/modules/audio_analyzer.py`

| Attribute | Value |
|-----------|-------|
| **Responsibility** | Extract prosodic and spectral audio features for behavioral analysis |
| **Inputs** | Audio file path (WAV/MP3/FLAC) or numpy array (microphone) |
| **Outputs** | `ProsodicFeatures` dataclass with 20+ feature fields |
| **Tools Called** | librosa, sounddevice, HFModelManager (optional wav2vec2) |
| **Quality Gate** | RMS > 0.001, duration > 0.5s, spectral_centroid 50–8000 Hz |

**Key Methods:**
- `analyze_file(path: str) -> ProsodicFeatures`
- `analyze_microphone(duration: int) -> ProsodicFeatures`
- `to_feature_vector(features: ProsodicFeatures) -> np.ndarray`
- `validate_audio(path: str) -> bool`
- `_detect_hedging_pauses(y, sr) -> bool`
- `_detect_voice_rise(f0) -> bool`
- `_detect_emotional_stress(f0) -> bool`

### Module 2: `agent/modules/visual_analyzer.py`

| Attribute | Value |
|-----------|-------|
| **Responsibility** | Facial action unit detection and expression classification from video |
| **Inputs** | Image path, video path, or numpy frame array |
| **Outputs** | `VisualFeatures` dataclass with AU scores, expression label, microexpression flag |
| **Tools Called** | MediaPipe FaceMesh, OpenCV |
| **Quality Gate** | Face detection confidence > 0.5; at least 300 landmarks detected |

**Key Methods:**
- `analyze_image(path: str) -> VisualFeatures`
- `analyze_video_frame(frame: np.ndarray) -> VisualFeatures`
- `analyze_video_file(path: str) -> List[VisualFeatures]`
- `is_available() -> bool`
- `_compute_action_units(landmarks) -> dict`
- `_detect_microexpression(prev_aus, curr_aus, elapsed_ms) -> bool`

### Module 3: `agent/modules/behavior_classifier.py`

| Attribute | Value |
|-----------|-------|
| **Responsibility** | Apply Gottman/Attachment/Love Language frameworks to multimodal features |
| **Inputs** | `ProsodicFeatures`, text string, optional `VisualFeatures` |
| **Outputs** | `ClassificationResult` dataclass with all framework scores |
| **Tools Called** | HFModelManager (emotion-distilroberta), regex pattern matching |
| **Quality Gate** | At least one modality must score above noise threshold (0.1) |

**Key Methods:**
- `classify(audio, text, visual) -> ClassificationResult`
- `_score_gottman_horsemen(text, audio, visual) -> dict`
- `_detect_attachment_pattern(audio, text) -> tuple[str, float]`
- `_detect_love_languages(text) -> dict`
- `_detect_non_literal(text, audio) -> tuple[bool, str, str]`
- `_compute_distress_score(horsemen, attachment, visual) -> float`

### Module 4: `agent/modules/interpretation_engine.py`

| Attribute | Value |
|-----------|-------|
| **Responsibility** | LLM synthesis of multimodal classification into empathy report |
| **Inputs** | `ClassificationResult`, `ProsodicFeatures`, text, optional `VisualFeatures` |
| **Outputs** | `InterpretationResult` dataclass with empathy report + counseling recommendation |
| **Tools Called** | LLMClient (Claude/OpenAI/Ollama), HFModelManager (bge-large RAG), MemoryManager |
| **Quality Gate** | LLM response must parse to valid JSON; fallback dict used if parsing fails |

**Key Methods:**
- `interpret(classification, audio, text, visual) -> InterpretationResult`
- `format_result(result: InterpretationResult) -> str`
- `_build_empathy_prompt(classification, audio, text, visual) -> str`
- `_build_counseling_prompt(classification) -> str`
- `_parse_llm_response(response_text) -> dict`

---

## HuggingFace Model Selection

| Model ID | Task | Benchmark Score | Chosen Over |
|----------|------|----------------|-------------|
| `facebook/wav2vec2-base` | Speech embedding | SUPERB: 94.3% phone acc | Mel-spectrogram CNN (no pre-training), SpeechBERT (heavier) |
| `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion (text) | F1: 0.66 macro GoEmotions | cardiffnlp/twitter-roberta-base-emotion (fewer classes) |
| `openai/whisper-large-v3` | ASR + word timestamps | WER: 2.7% LibriSpeech clean | whisper-medium (lower accuracy), DeepSpeech (outdated) |
| `BAAI/bge-large-en-v1.5` | Text embedding / RAG | MTEB: 64.23 avg | OpenAI text-embedding-ada-002 (API cost), all-MiniLM (lower quality) |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | BEIR nDCG@10: 0.491 | bge-reranker-base (lower accuracy), monot5 (slower) |
| `facebook/bart-large-cnn` | Summarization | ROUGE-2: 21.28 CNN-DM | pegasus (slower), distilbart (lower quality) |

---

## LLM API Integration Spec

### Provider Chain
```
Claude (claude-opus-4-8) → OpenAI (gpt-4o) → Ollama (llama3)
```

### Prompt Templates

**1. EMPATHY_REPORT_PROMPT (System)**
```
You are a compassionate relationship science expert trained in Gottman Method Couples Therapy,
Attachment Theory (Ainsworth), and the Five Love Languages (Chapman). Your role is to help users
understand their partner's emotional state — NOT to manipulate or control.

Analyze the following multimodal data and produce a structured empathy report in JSON format:
{multimodal_context}

Respond ONLY with valid JSON matching this schema:
{
  "actual_feeling": "string (what partner is actually feeling)",
  "stated_vs_actual_gap": "string (explain the incongruence if any)",
  "gottman_pattern_active": "string (which Horseman or 'none')",
  "attachment_context": "string (attachment theory explanation)",
  "love_language_expressed": "string (which love language is active)",
  "empathy_responses": ["string", "string", "string"],
  "urgency_level": "low|medium|high|counseling_needed",
  "counseling_recommended": boolean,
  "counseling_message": "string (only if counseling_recommended=true)",
  "confidence_explanation": "string",
  "supporting_research": ["citation1", "citation2"]
}

SAFETY RULES:
- Never suggest specific actions the user should take that could be harmful
- Frame all suggestions as optional empathy options, never commands
- If counseling_recommended=true, always include a supportive counseling_message
- Do not diagnose clinical conditions
```

**2. HEDGING_ANALYSIS_PROMPT**
```
Identify hedging and indirect language patterns in: "{text}"
Return JSON: {"hedging_detected": bool, "phrases": [], "literal_meaning": "", "probable_actual_meaning": ""}
```

**3. GOTTMAN_ANALYSIS_PROMPT**
```
Apply Gottman's Four Horsemen framework to these communication features: {features}
Score each horseman 0.0–1.0 and explain the linguistic evidence.
```

**4. COUNSELING_RECOMMENDATION_PROMPT**
```
The communication analysis has detected high-conflict patterns (distress_score={score}).
Generate a compassionate, non-alarmist recommendation for professional couples counseling.
Emphasize: it is a sign of strength, not failure, to seek professional support.
```

### Token Budget Estimates
- EMPATHY_REPORT_PROMPT: ~500 input tokens + ~600 output tokens = ~1100 tokens / session
- At claude-opus-4-8 pricing: ~$0.033 per session
- Monthly (100 sessions/day): ~$99/month → use claude-sonnet-4-6 for cost reduction

---

## End-to-End Execution Flow

1. **User Input:** User invokes CLI `analyze --audio partner.wav` or REST `POST /api/v1/analyze/audio`
2. **Audio Validation:** `AudioAnalyzer.validate_audio(path)` — 3-gate check (RMS, duration, spectral range)
3. **Transcription:** `HFModelManager.transcribe(audio_path)` → text + word timestamps (Whisper)
4. **Prosodic Features:** `AudioAnalyzer.analyze_file(path)` → `ProsodicFeatures` (MFCC, F0, speech rate, pauses)
5. **Emotion Classification:** `HFModelManager.classify_emotion(text)` → 7-class scores
6. **Visual Analysis:** If video provided → `VisualAnalyzer.analyze_video_file(path)` → `List[VisualFeatures]`
7. **Behavior Classification:** `BehaviorClassifier.classify(audio, text, visual)` → `ClassificationResult`
8. **Safety Gate:** If `distress_score > 0.8` or any horseman > 0.7 → set `needs_counseling_flag=True`
9. **LLM Interpretation:** `InterpretationEngine.interpret(classification, audio, text, visual)` → `InterpretationResult`
10. **Memory + Output:** `MemoryManager.save_session(...)` → return JSON + formatted ASCII report

**Error Handling per Step:**
- Step 2: Invalid audio → return `AudioValidationError` with specific gate failure message
- Step 3: Whisper unavailable → use text input directly or `TRANSCRIPTION_FAILED` marker
- Steps 4–6: Model unavailable → use handcrafted fallback features (text-only mode)
- Step 7: Classification failure → return neutral baseline classification with low confidence
- Step 9: All LLM providers fail → return `FALLBACK_INTERPRETATIONS[classification.communication_style]`
- Step 10: DB write failure → log error, return result without persisting

---

## SECOND-KNOWLEDGE-BRAIN.md Integration

- **RAG Query:** `InterpretationEngine` encodes classification summary with bge-large-en-v1.5, retrieves top-3 most relevant paper abstracts from SECOND-KNOWLEDGE-BRAIN.md to include as context in LLM prompt
- **Update Trigger:** `KnowledgeUpdater.run()` called by APScheduler every Sunday 02:00 local time
- **Dedup:** SHA256 hash of paper URL/DOI stored in `knowledge_hashes` SQLite table; skip if present
- **Impact:** Each new paper cycle adds ≥5 new research findings to LLM context window; agent improves continuously

---

## knowledge_updater.py Spec

| Attribute | Value |
|-----------|-------|
| **Schedule** | Weekly, Sunday 02:00 local |
| **Sources** | ArXiv (cs.CL, cs.CV, cs.HC), Semantic Scholar Graph API |
| **Queries** | 5 ArXiv + 5 Semantic Scholar queries (see file for full list) |
| **Max papers per run** | 50 (configurable) |
| **Scoring** | recency_score × 0.6 + relevance_score × 0.4 |
| **Failure handling** | Retry 3× with 30s backoff; log failure to SECOND-KNOWLEDGE-BRAIN.md; do NOT crash agent |

---

## llm_client.py Spec

| Attribute | Value |
|-----------|-------|
| **Provider chain** | Claude → OpenAI → Ollama |
| **Retry logic** | Exponential backoff: 1s, 2s, 4s (3 attempts per provider) |
| **Streaming** | Supported for all 3 providers |
| **Cost tracking** | Logged to `llm_cost_log` SQLite table per call |
| **PRIVACY_MODE** | env=true forces Ollama only; no data leaves local machine |

---

## hf_model_manager.py Spec

| Attribute | Value |
|-----------|-------|
| **Pattern** | Singleton per process |
| **Loading** | Lazy — download on first use |
| **Cache** | `./models/` directory (configurable via config) |
| **CUDA** | Auto-detect; fall back to CPU |
| **Idle unload** | 600-second timer resets on each use; model unloaded from RAM when idle |
| **Fallback** | TF-IDF for embeddings if bge-large unavailable |

---

## Docker Compose Spec

| Service | Purpose |
|---------|---------|
| `partner-decode-agent` | Main agent, CPU mode, port 8011 |
| `partner-decode-agent-gpu` | Main agent, GPU profile, NVIDIA CUDA passthrough |
| `ollama` | Local LLM server for offline/privacy mode, port 11434 |

**Volumes:** `partner_data` (SQLite DB + session files), `partner_models` (HF model cache), `partner_ollama_models` (Ollama models)

---

## Quality Gates

1. **Audio Quality Gate:** RMS energy > 0.001 AND duration > 0.5s AND spectral centroid in [50, 8000] Hz — reject silence/noise before processing
2. **Emotion Confidence Gate:** Emotion classification confidence > 0.3 — if all 7 classes below threshold, classify as "uncertain" rather than forcing a label
3. **Classification Confidence Gate:** BehaviorClassifier overall confidence > 0.2 before proceeding to LLM — prevents low-signal hallucinated reports
4. **Safety Gate:** `distress_score > 0.8` OR any Horseman score > 0.7 → `counseling_flag=True` AND `counseling_message` must be non-empty in output
5. **LLM Output Gate:** JSON parse must succeed OR fallback interpretations must be returned — never return raw unparsed LLM text to user
6. **Harm Prevention Gate:** `InterpretationEngine` scans output for prohibited phrases (specific medication/psychological diagnoses/manipulative suggestions) before returning
7. **Privacy Gate:** PRIVACY_MODE=true → all LLM calls route to Ollama only; verified at startup with assertion; agent refuses to start if Ollama unreachable in privacy mode

---

## Test Scenarios

1. **Non-literal Expression (Golden Path):** Audio "I'm fine, don't worry" with high F0 variance + fast speech rate → `non_literal_detected=True`, `actual_emotional_state=stressed`, empathy report identifies gap
2. **Gottman Contempt Detection:** Text "Of course YOU would think that" + monotone dismissive audio → `contempt score > 0.7`, `urgency_level=high`, counseling flag set
3. **Anxious Attachment:** Repeated short audio segments with rising intonation + reassurance-seeking text → `attachment_pattern=anxious`, correct love language signals (words_of_affirmation)
4. **Stonewalling Detection:** Very low RMS audio + minimal text ("ok", "fine", "whatever") → `stonewalling score > 0.6`, `overall_distress_score > 0.7`
5. **Positive Secure Interaction:** Warm calm voice + affirmative text → `attachment_pattern=secure`, `gottman_horsemen all < 0.2`, `urgency_level=low`
6. **Knowledge Crawler Deduplication:** Run `update_knowledge()` twice on same day → second run adds 0 new papers (SHA256 dedup verified)
7. **All LLM Providers Down:** Mock all 3 providers to raise exceptions → `FALLBACK_INTERPRETATIONS` returned with `counseling_recommended=False` and note about offline mode
8. **REST API Full Integration:** `POST /api/v1/analyze/session` with audio bytes + text → HTTP 200, valid JSON response with all required fields populated

---

## Key Design Decisions

1. **Multimodal fusion is late-stage:** Each modality (audio, visual, text) is processed independently before fusion at BehaviorClassifier level. This allows graceful degradation when one modality is unavailable — the agent still produces a report, clearly noting which modalities were available.

2. **No model training from scratch:** All ML inference uses HuggingFace pretrained models (wav2vec2, emotion-distilroberta, whisper). This makes the agent deployable on consumer hardware without GPU clusters.

3. **LLM as final reasoning layer only:** The LLM (Claude) is NOT used for raw feature extraction or classification. It is invoked only after all quantitative features are computed, reducing hallucination risk and keeping LLM calls to one per session.

4. **Ethical framing is non-negotiable:** The agent never positions itself as a "spy" or "manipulation tool." All outputs use language like "your partner may be expressing..." rather than "your partner is hiding...". The empathy_responses field contains suggestions, not scripts.

5. **Safety gates are hard-coded, not configurable:** The `needs_counseling_flag` trigger thresholds (0.8 distress, 0.7 horseman) are hardcoded and cannot be disabled via config. This prevents the safety system from being misconfigured out of existence.

6. **SQLite WAL mode for reliability:** WAL (Write-Ahead Logging) mode allows concurrent reads during writes, which is critical when the background APScheduler knowledge updater thread writes while the web server handles requests.

7. **MediaPipe is optional:** The visual analysis layer degrades gracefully. If MediaPipe cannot be installed (e.g., ARM systems, minimal Docker), `VisualAnalyzer.is_available()` returns False and all visual fields are `None`. The agent still provides a complete audio + text report.
