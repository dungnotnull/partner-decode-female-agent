<div align="center">

# 🧠 partner-decode-female-agent

**Multimodal AI decoder for understanding female partner communication**

*Audio prosody • Text sentiment • Facial expressions • Relationship science*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

<p align="center">
  <em>When she says "I'm fine" with a tense voice — <br>
  this agent decodes what she's actually feeling,<br>
  using validated relationship science frameworks.</em>
</p>

</div>

---

## ✨ What It Does

partner-decode-female-agent analyzes **voice audio**, **text messages**, and **optional video** through the lens of three validated relationship-science frameworks to surface the emotional truth behind expressed communication:

| Framework | What It Detects | Source |
|-----------|---------------|--------|
| 🔴 **Gottman Four Horsemen** | Criticism, contempt, defensiveness, stonewalling | Gottman (1994) — 93.6% divorce prediction |
| 🟡 **Attachment Theory** | Anxious, avoidant, secure attachment patterns | Ainsworth (1978) — Strange Situation paradigm |
| 🟢 **Five Love Languages** | Words of affirmation, acts of service, gifts, quality time, physical touch | Chapman (1992) — 35 years of counseling |

### Key Capabilities

- 🎙️ **Audio Prosody**: MFCC (40+coefficients), F0 contour (female range 75–400 Hz), speech rate, pause detection, hedging detection, emotional stress markers
- 📝 **Text Analysis**: Gottman pattern regex (60+ patterns), attachment classification, love language signals, 20 non-literal expression decoders ("I'm fine" ≠ fine)
- 👁️ **Visual (Optional)**: MediaPipe FaceMesh 468-landmark analysis, 9 facial action units, microexpression detection (<200ms), blink rate, gaze direction
- 🤖 **6 HuggingFace Models**: wav2vec2, emotion-distilroberta, whisper-large-v3, bge-large, bge-reranker, bart-cnn
- 🧠 **3 LLM Providers**: Claude (primary) → OpenAI (fallback) → Ollama (offline/privacy)
- 🛡️ **7 Quality Gates**: Audio validation, emotion confidence, classification confidence, safety, LLM output, harm prevention, privacy
- 💾 **Persistent Memory**: SQLite WAL with 5 tables, partner profile tracking, cost logging
- 📊 **Prometheus Metrics**: Sessions, errors, costs, latency
- 🔬 **Self-Improving**: Weekly ArXiv + Semantic Scholar knowledge crawl with deduplication

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    partner-decode-female-agent                   │
│                                                                 │
│  Input Layer        Feature Extraction       Classification      │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Audio    │  │AudioAnalyzer│  │ HFModelMgr   │  │  Behavior │ │
│  │ Text     │→ │VisualAnalyz│→ │ (6 models)   │→ │Classifier│ │
│  │ Video    │  │             │  │              │  │           │ │
│  │ Microphone│  │             │  │              │  │           │ │
│  └─────────┘  └─────────────┘  └──────────────┘  └─────┬─────┘ │
│                                                          │      │
│                     ┌────────────────────────────────────┘      │
│                     │  Safety Gate (distress > 0.8)             │
│                     ▼                                            │
│              ┌──────────────────┐  ┌─────────────┐             │
│              │InterpretationEng.│← │  LLMClient   │             │
│              │ (Claude/OpenAI/  │  │ (3 providers)│             │
│              │  Ollama)        │  │              │             │
│              └───────┬──────────┘  └─────────────┘             │
│                      │                                         │
│              ┌───────▼──────────┐                               │
│              │  MemoryManager   │                               │
│              │  (SQLite WAL)    │                               │
│              └──────────────────┘                               │
│                     │  Output                                   │
│         JSON + ASCII empathy report + urgency level             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/dungnotnull/partner-decode-female-agent.git
cd partner-decode-female-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

### CLI Usage

```bash
# Decode a text message
python agent/main.py decode "I'm fine, don't worry about me"

# Analyze an audio file
python agent/main.py analyze partner_voice.wav

# Full multimodal session (audio + text + video)
python agent/main.py session --audio audio.wav --text "whatever" --video video.mp4

# Record from microphone (30 seconds)
python agent/main.py listen --duration 30

# View session history
python agent/main.py history --n 10

# Update knowledge base (crawl ArXiv + Semantic Scholar)
python agent/main.py update-knowledge

# View LLM cost report
python agent/main.py cost-report

# Start REST API server
python agent/main.py serve --host 0.0.0.0 --port 8011
```

### REST API

```bash
# Health check
curl http://localhost:8011/health

# Decode text
curl -X POST http://localhost:8011/api/v1/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "I'm fine, whatever"}'

# Analyze audio
curl -X POST http://localhost:8011/api/v1/analyze/audio \
  -F "audio_file=@partner_voice.wav"

# Full multimodal session
curl -X POST http://localhost:8011/api/v1/analyze/session \
  -F "audio_file=@audio.wav" \
  -F "text=I'm okay, really" \
  -F "video_file=@video.mp4"

# Session history
curl http://localhost:8011/api/v1/sessions?n=5

# LLM cost breakdown
curl http://localhost:8011/api/v1/cost

# Statistics
curl http://localhost:8011/api/v1/stats

# Prometheus metrics
curl http://localhost:8011/metrics
```

### Docker

```bash
# CPU mode
docker compose -f docker/docker-compose.yml --profile cpu up -d

# GPU mode (requires NVIDIA Docker)
docker compose -f docker/docker-compose.yml --profile gpu up -d

# Privacy mode (Ollama only — no cloud APIs)
docker compose -f docker/docker-compose.yml --profile privacy up -d
```

---

## 🔬 Science Frameworks

### Gottman's Four Horsemen

| Horseman | Detection | Linguistic Markers |
|----------|-----------|-------------------|
| **Criticism** | 20 regex patterns | "you always", "you never", "you're so" |
| **Contempt** | 15 regex patterns + visual AU boost | "that's ridiculous", eye-roll AU7+AU9 |
| **Defensiveness** | 15 regex patterns | "but I", "it's not my fault", "yes, but" |
| **Stonewalling** | 10 regex patterns + low-RMS audio boost | "fine", "whatever", "ok" + flat prosody |

### Attachment Theory (Ainsworth)

| Style | Prosodic Markers | Linguistic Markers |
|--------|-----------------|-------------------|
| **Anxious** | High F0 variance (>40 Hz), fast speech (>4 syl/s), rising intonation | "Do you still love me?", reassurance-seeking |
| **Avoidant** | Flat F0 contour (<10 Hz std), low energy, slow speech | "I need space", "I'm fine" (actually not fine) |
| **Secure** | Moderate F0 variance (10–40 Hz), few pauses | "I understand", "we can figure this out" |

### Non-Literal Expression Decoder

| Phrase | Prosodic Cue | Actual Meaning |
|--------|-------------|---------------|
| "I'm fine" | Stressed prosody | I am not fine and need acknowledgment |
| "Whatever" | Rising F0 | I feel dismissed and am withdrawing |
| "I don't care" | Stressed prosody | I care deeply but feel unable to say so |
| "Never mind" | High energy | I want you to ask what is wrong |
| "Just go" | High F0 variance | I want you to choose to stay |

*20 non-literal expressions in total — see `behavior_classifier.py` for the full list*

---

## 🛡️ Safety Gates

The agent includes **7 hard-coded safety gates** that cannot be disabled via configuration:

| Gate | Threshold | Action |
|------|-----------|--------|
| 1. Audio Quality | RMS > 0.001, duration > 0.5s, spectral centroid 50–8000 Hz | Reject invalid audio |
| 2. Emotion Confidence | All emotion scores < 0.3 | Label as "uncertain" |
| 3. Classification Confidence | Overall confidence < 0.2 | Add reliability warning |
| 4. Safety | distress > 0.8 OR any Horseman > 0.7 | Set counseling_flag = True |
| 5. LLM Output | JSON parse must succeed | Use fallback interpretations |
| 6. Harm Prevention | Prohibited pattern scan | Replace with safe fallback |
| 7. Privacy | `PRIVACY_MODE=true` | Force Ollama only — zero bytes to cloud |

---

## 📁 Project Structure

```
partner-decode-female-agent/
├── agent/
│   ├── __init__.py                    # Package exports
│   ├── main.py                        # CLI + FastAPI REST server (8 commands, 10 endpoints)
│   ├── orchestrator.py                # PartnerDecodeOrchestrator + quality gates
│   ├── modules/
│   │   ├── __init__.py                # Module exports
│   │   ├── audio_analyzer.py          # ProsodicFeatures (28 fields) + AudioAnalyzer
│   │   ├── visual_analyzer.py         # VisualFeatures (12 fields) + VisualAnalyzer
│   │   ├── behavior_classifier.py     # ClassificationResult (13 fields) + BehaviorClassifier
│   │   └── interpretation_engine.py    # InterpretationResult (12 fields) + InterpretationEngine
│   └── memory/
│       ├── __init__.py                # Memory exports
│       └── memory_manager.py          # SQLite WAL, 5 tables, thread-safe
├── tools/
│   ├── __init__.py                    # Tools exports
│   ├── hf_model_manager.py           # 6 HF models, lazy load, 600s idle unload
│   ├── llm_client.py                  # Claude → OpenAI → Ollama, streaming, cost tracking
│   └── knowledge_updater.py           # ArXiv + Semantic Scholar weekly crawl
├── config/
│   ├── agent_config.yaml             # Runtime configuration (14 sections)
│   └── .env.example                   # Environment variables template
├── docker/
│   ├── Dockerfile                     # python:3.12-slim, non-root, healthcheck
│   └── docker-compose.yml             # 3 services: agent, agent-gpu, ollama
├── tests/
│   ├── test_agent.py                  # 51 test methods (mocked, no real API calls)
│   └── test-scenarios.md              # 8 end-to-end scenarios
├── CLAUDE.md                          # Agent identity card
├── PROJECT-detail.md                  # Full technical specification
├── PROJECT-DEVELOPMENT-PHASE-TRACKING.md  # 194/194 tasks done
├── SECOND-KNOWLEDGE-BRAIN.md          # Self-improving knowledge base (15+ papers)
├── requirements.txt                   # Pinned dependencies
└── README.md                          # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_agent.py -v

# Run specific test class
pytest tests/test_agent.py::TestBehaviorClassifier -v

# Run with coverage
pytest tests/test_agent.py --cov=agent --cov=tools -v
```

**Test coverage:** 51 test methods across 9 test classes:

| Test Class | Tests | What It Verifies |
|-----------|-------|-----------------|
| `TestAudioAnalyzer` | 7 | MFCC, F0, speech rate, validation |
| `TestVisualAnalyzer` | 7 | AU computation, microexpression, graceful None |
| `TestBehaviorClassifier` | 8 | Horsemen, attachment, love languages, safety gate |
| `TestInterpretationEngine` | 7 | LLM synthesis, fallback, JSON parsing, harm scan |
| `TestMemoryManager` | 6 | SQLite WAL, thread safety, cost logging, dedup |
| `TestLLMClient` | 3 | Provider fallback, PRIVACY_MODE, cost computation |
| `TestHFModelManager` | 3 | Singleton, emotion fallback, TF-IDF fallback |
| `TestIntegrationPipeline` | 5 | Full pipeline, report structure, LLM fallback, memory |
| `TestCLISmokeTests` | 5 | CLI --help, version, all commands |

All tests use `unittest.mock` — **zero real API calls** are made.

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude API key (primary LLM) |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback LLM) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (offline LLM) |
| `PRIVACY_MODE` | `false` | Force Ollama only — no cloud APIs |
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8011` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `USE_WAV2VEC2` | `true` | Enable wav2vec2 embeddings |
| `USE_VISUAL_ANALYSIS` | `true` | Enable MediaPipe visual analysis |

### LLM Provider Priority

```
Claude (claude-opus-4-8) → OpenAI (gpt-4o) → Ollama (llama3)
```

Set `PRIVACY_MODE=true` to force Ollama only. The agent will refuse to start if Ollama is unreachable in privacy mode.

---

## 📊 HuggingFace Models

| Model | Task | Size | When Loaded |
|-------|------|------|------------|
| `facebook/wav2vec2-base` | Speech embedding (768-dim) | ~360MB | First audio analysis |
| `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion (text) | ~270MB | First text classification |
| `openai/whisper-large-v3` | ASR + word timestamps | ~3GB | First audio transcription |
| `BAAI/bge-large-en-v1.5` | Text embedding (RAG) | ~1.3GB | First knowledge retrieval |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | ~1.3GB | First knowledge retrieval |
| `facebook/bart-large-cnn` | Summarization | ~1.6GB | First session summarization |

All models use **lazy loading** (downloaded on first use) and **600-second idle unload** to conserve RAM. CUDA is auto-detected with CPU fallback.

---

## 🔒 Privacy & Ethics

This agent is designed as an **empathy and self-improvement tool**, never a surveillance or manipulation tool:

- **Safety gates are hard-coded** — counseling thresholds (0.8 distress, 0.7 horseman) cannot be disabled
- **Harm prevention scan** — all LLM output is scanned for prohibited patterns (medication suggestions, diagnoses, manipulation instructions)
- **PRIVACY_MODE** — forces all processing to local Ollama; zero bytes sent to cloud APIs
- **Counseling recommendation** — always presented as an option, never a command; emphasizes seeking professional help as a sign of strength
- **Empathy framing** — all responses say "your partner may be expressing..." not "your partner is..."

---

## 📖 Knowledge Base

The `SECOND-KNOWLEDGE-BRAIN.md` is a self-improving knowledge base that starts with 15 foundational papers and automatically grows weekly:

- **ArXiv** (cs.CL, cs.CV, cs.HC) — 5 queries per category
- **Semantic Scholar** — 5 relationship science queries
- **Scoring**: recency × 0.6 + relevance × 0.4
- **Deduplication**: SHA256 hash of paper URL/DOI
- **Update schedule**: Every Sunday 02:00 via APScheduler

---

## 📈 Cost Estimation

| Provider | Model | Input/1K tokens | Output/1K tokens | Est. cost/session |
|----------|-------|-----------------|-------------------|-------------------|
| Anthropic | claude-opus-4-8 | $0.015 | $0.075 | ~$0.033 |
| OpenAI | gpt-4o | $0.005 | $0.015 | ~$0.011 |
| Ollama | llama3 | Free | Free | $0.00 |

Cost is tracked per call in the `llm_cost_log` table and accessible via `/api/v1/cost`.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all existing tests pass (`pytest tests/test_agent.py -v`)
5. Commit with descriptive messages
6. Push to your fork and open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

**Scientific Foundations:**
- John Gottman — Four Horsemen framework (Gottman, 1994)
- Mary Ainsworth — Attachment theory (Ainsworth, 1978)
- Gary Chapman — Five Love Languages (Chapman, 1992)
- Paul Ekman — Facial Action Coding System (Ekman, 2003)
- Alexei Baevski — wav2vec2 self-supervised speech (Baevski et al., 2020)
- Jared Hartmann — emotion-distilroberta (Hartmann, 2022)
- Camillo Lugaresi — MediaPipe FaceMesh (Lugaresi et al., 2019)

**Open Source Models:**
- Facebook AI — wav2vec2-base, bart-large-cnn
- OpenAI — whisper-large-v3
- j-hartmann — emotion-english-distilroberta-base
- BAAI — bge-large-en-v1.5, bge-reranker-large

---

<div align="center">

**Built with ❤️ for understanding, not manipulation.**

*If this tool reveals concerning patterns in your relationship, please seek professional couples counseling.*

</div>
