# CLAUDE.md — partner-decode-female-agent

## Agent Identity

**Name:** partner-decode-female-agent
**Tagline:** Understand your partner better through multimodal emotion and communication decoding
**Current Build Phase:** Phase 0 — Research & Architecture
**Version:** 1.0.0
**Cluster:** G — Multimodal Behavior & Emotion AI

---

## Problem Statement

Modern relationships suffer from communication gaps where what is said and what is meant diverge significantly. When a partner says "I'm fine" with a tense voice, or "do whatever you want" with flat affect, the literal words convey the opposite of the emotional truth. This agent decodes female partner communication by analyzing voice audio (prosodic features: pitch, speech rate, pauses, energy), text transcripts (sentiment, hedging language, Gottman patterns), and optional video (facial action units via MediaPipe FaceMesh 468 landmarks) to surface the actual emotional intent behind expressed communication. Using three validated relationship-science frameworks — Gottman's Four Horsemen, Attachment Theory (Ainsworth), and the Five Love Languages (Chapman) — the agent produces structured empathy reports that help the user understand, not manipulate, their partner. All analysis is framed as an educational self-improvement tool; the agent recommends professional couples counseling whenever high-conflict patterns are detected.

---

## Architecture Summary

1. **Input Ingestion:** User provides audio file, microphone stream, text message, or full multimodal session (audio + text + optional video)
2. **Audio Feature Extraction (AudioAnalyzer):** Extract MFCC (40 coefficients + delta + delta2), chroma, spectral features, F0 contour, speech rate, pause patterns, and optional wav2vec2 768-dim embeddings
3. **Visual Feature Extraction (VisualAnalyzer):** If video provided, MediaPipe FaceMesh detects 468 landmarks, computes facial action units (AU1/AU2/AU4/AU6/AU12/AU15/AU17/AU20/AU26), eye blink rate, gaze direction, and flags microexpressions (<200ms AU transitions)
4. **Whisper Transcription:** If audio provided without text, openai/whisper-large-v3 transcribes speech with word-level timestamps
5. **Emotion Classification (HFModelManager):** j-hartmann/emotion-english-distilroberta-base classifies text into 7 emotion scores (anger/disgust/fear/joy/neutral/sadness/surprise)
6. **Behavior Classification (BehaviorClassifier):** Gottman Four Horsemen detection, Attachment Pattern classification (anxious/avoidant/secure), Five Love Language signal detection, non-literal expression decoder
7. **Safety Gate Check:** If distress_score > 0.8 or any Horseman > 0.7, set counseling_flag=True
8. **LLM Interpretation (InterpretationEngine):** Claude (primary) synthesizes all features into structured empathy report with actionable responses
9. **Memory Persistence (MemoryManager):** Store session, update partner profile, log running communication pattern trends
10. **Report Output:** Structured JSON + formatted ASCII display with urgency level and optional counseling recommendation

---

## Module List (`agent/modules/`)

| File | Responsibility |
|------|---------------|
| `audio_analyzer.py` | Real-time MFCC/prosodic feature extraction + optional wav2vec2 embeddings; F0, speech rate, pause detection; vocal cue classification |
| `visual_analyzer.py` | MediaPipe FaceMesh 468-landmark analysis; facial action unit computation; microexpression detection; expression classification |
| `behavior_classifier.py` | Gottman Four Horsemen scoring; Attachment Theory pattern detection; Five Love Language signal detection; non-literal expression decoder |
| `interpretation_engine.py` | LLM-powered empathy report synthesis; Gottman/Attachment/Love Language framework application; counseling recommendation generation |

---

## Tools List (`agent/tools/` within agent/)

The agent also calls the top-level `tools/` utilities below.

---

## Supporting Tools (`tools/`)

| File | Responsibility |
|------|---------------|
| `knowledge_updater.py` | crawl4ai + ArXiv API + Semantic Scholar pipeline; weekly cron to update SECOND-KNOWLEDGE-BRAIN.md with latest relationship science and affective computing papers |
| `llm_client.py` | Unified LLM client: Claude (primary) → OpenAI (fallback) → Ollama (offline); exponential backoff retry; streaming support; cost logging |
| `hf_model_manager.py` | Singleton HuggingFace model registry; lazy loading; CUDA auto-detect; 600s idle unload; models: wav2vec2, emotion-distilroberta, whisper-large-v3, bge-large, bge-reranker, bart-cnn |

---

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|-----------|
| `facebook/wav2vec2-base` | Prosodic speech embedding (768-dim) | Self-supervised speech representation; captures subtle vocal tone cues without labeled data |
| `j-hartmann/emotion-english-distilroberta-base` | 7-class text emotion classification | Highest accuracy on GoEmotions benchmark for relationship-relevant emotion labels; lightweight distilled model |
| `openai/whisper-large-v3` | Speech-to-text transcription with word timestamps | Best-in-class multilingual ASR; word-level timestamps enable prosody-text alignment |
| `pyannote/speaker-diarization-3.1` | Speaker diarization (optional multi-speaker) | Identifies who speaks when; enables isolating partner's voice in shared audio |
| `BAAI/bge-large-en-v1.5` | Text embeddings for RAG (knowledge base retrieval) | Top MTEB leaderboard score; used for retrieving relevant research from SECOND-KNOWLEDGE-BRAIN.md |
| `facebook/bart-large-cnn` | Summarization of long session transcripts | CNN-DM fine-tuned; reliable extractive-abstractive summarization for session history |

---

## LLM API Integration

| Provider | Model | Use Case |
|----------|-------|---------|
| **Claude (primary)** | claude-opus-4-8 | Long-context empathy report synthesis; Gottman/Attachment/Love Language framework reasoning; nuanced non-literal expression interpretation; counseling recommendation generation |
| **OpenAI (fallback)** | gpt-4o | Multimodal tasks when video frames need LLM-level analysis; structured JSON function calling for emotion output |
| **Ollama (offline)** | llama3 | Privacy-sensitive sessions where user does not want audio/text sent to cloud; high-volume offline mode |

**Provider Priority:** claude → openai → ollama (PRIVACY_MODE env forces Ollama)

---

## Knowledge Crawl Sources

| Source | Categories / Queries | Update Frequency |
|--------|---------------------|-----------------|
| ArXiv API | cs.CL (NLP/communication), cs.CV (facial expression), cs.HC (human-computer interaction) | Weekly (Sunday 02:00) |
| Semantic Scholar Graph API | Relationship communication, attachment theory NLP, multimodal emotion recognition, Gottman patterns | Weekly |
| Journal of Personality and Social Psychology | Relationship science, emotion regulation, attachment patterns | Weekly |
| Communication Research (SAGE) | Couple communication patterns, hedging language, emotional expression | Weekly |
| Gottman Institute Publications | Four Horsemen research, relationship prediction studies | Weekly |
| Papers with Code | Emotion recognition leaderboards, prosody analysis SOTA | Weekly |

---

## Active Development Tasks

- [ ] Phase 0: Research & Architecture — read upstream literature, define module boundaries
- [ ] Phase 1: AudioAnalyzer — MFCC + prosodic features + wav2vec2 integration
- [ ] Phase 1: VisualAnalyzer — MediaPipe FaceMesh + AU computation + microexpression detection
- [ ] Phase 1: BehaviorClassifier — Four Horsemen + Attachment + Love Language + non-literal decoder
- [ ] Phase 1: InterpretationEngine — LLM empathy report synthesis + safety gates
- [ ] Phase 2: Orchestrator — multimodal pipeline wiring + session management
- [ ] Phase 2: MemoryManager — SQLite WAL + 5 tables + partner profile
- [ ] Phase 3: HFModelManager — all 6 models lazy-loaded with CUDA auto-detect
- [ ] Phase 4: LLMClient — Claude/OpenAI/Ollama with retry + streaming
- [ ] Phase 5: KnowledgeUpdater — ArXiv + Semantic Scholar weekly crawl
- [ ] Phase 5: First SECOND-KNOWLEDGE-BRAIN.md crawl run
- [ ] Phase 6: Docker Compose + Dockerfile
- [ ] Phase 6: Full test suite (44 tests) passing
- [ ] Phase 7: REST API endpoints (FastAPI + Prometheus metrics)
- [ ] Phase 7: CLI interface (Click: analyze/listen/decode/session/serve)
