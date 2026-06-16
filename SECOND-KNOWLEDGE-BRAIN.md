# SECOND-KNOWLEDGE-BRAIN.md — partner-decode-female-agent

> Self-improving knowledge base. Updated weekly by `tools/knowledge_updater.py`.
> The longer this agent runs, the more accurate and capable it becomes.

---

## Core Concepts & Frameworks

### Gottman's Four Horsemen of the Apocalypse
Developed by Dr. John Gottman at the University of Washington after observing thousands of couples. These four communication behaviors predict relationship dissolution with 93.6% accuracy (Gottman & Levenson, 1992):

1. **Criticism** — attacking the partner's character or personality rather than a specific behavior. Distinguished from complaint (specific behavioral feedback) by its global nature. Linguistic markers: "you always", "you never", "you are so...", character-level attacks.

2. **Contempt** — treating the partner as inferior; the single strongest predictor of divorce. Includes sarcasm, cynicism, mockery, eye-rolling, sneering, hostile humor. Rooted in long-simmering negative thoughts. Linguistic markers: "that's ridiculous", "you don't even know", dismissive tone, dramatic eye-roll (AU7+AU9+AU45).

3. **Defensiveness** — denying responsibility, making excuses, counter-attacking. Typically a response to criticism but escalates conflict. Linguistic markers: "it's not my fault", "yes but", "you're wrong, I didn't...", justification chains.

4. **Stonewalling** — emotional withdrawal from interaction; shutting down communication. Physiological flooding (heart rate > 100 BPM) often precedes stonewalling. Behavioral markers: monosyllabic responses, turning away, minimal vocalizations, very low vocal energy.

**The Positive Sentiment Override (PSO)** — couples with high PSO interpret neutral behaviors as positive; those with Negative Sentiment Override (NSO) interpret neutral behaviors negatively. This context dramatically affects how the Four Horsemen manifest.

### Attachment Theory
Developed by John Bowlby (1969) and empirically tested by Mary Ainsworth (1978) using the Strange Situation paradigm. Three primary attachment styles (Ainsworth) extended to four by Main & Solomon (1990):

- **Secure Attachment:** Comfortable with intimacy and independence. Seeks support when distressed, provides support effectively. Linguistic markers: direct communication, acknowledgment, "we" orientation.

- **Anxious (Preoccupied) Attachment:** Hyperactivates attachment system. Seeks excessive reassurance, fears abandonment, high emotional reactivity. Linguistic markers: "do you still love me?", repeated contact attempts, "you never...", rumination patterns. Prosodic markers: high F0 variance, fast speech rate, rising intonation at statement ends.

- **Avoidant (Dismissing) Attachment:** Deactivates attachment system. Minimizes intimacy, values independence to extreme, emotional suppression. Linguistic markers: very short responses, deflection, "I'm fine" (actually not fine), topic changes. Prosodic markers: flat F0 contour, low energy, slow speech rate.

- **Disorganized (Fearful) Attachment:** Both desires and fears intimacy; associated with unresolved trauma. Inconsistent behavioral patterns.

### Five Love Languages
Developed by Gary Chapman (1992) from 35 years of couples counseling. Each person has a primary and secondary love language through which they express and best receive love:

1. **Words of Affirmation** — verbal expressions of love, appreciation, encouragement. Linguistic markers: "I love you", "you did a great job", "thank you for...", verbal praise patterns.

2. **Acts of Service** — doing helpful things for the partner. Linguistic markers: requests for specific help, appreciation of assistance, "could you please...", service-oriented expressions.

3. **Receiving Gifts** — meaningful physical tokens of care. Linguistic markers: "I saw this and thought of you", gift references, material appreciation expressions.

4. **Quality Time** — undivided attention and shared experiences. Linguistic markers: "we should spend more time...", "I miss us", togetherness requests, "we" language frequency.

5. **Physical Touch** — physical affection and presence. Linguistic markers: references to physical contact, requests for hugging/touching, tactile expression vocabulary.

### Communication Science Frameworks
**Congruence Theory (Satir, 1972):** Communication is congruent when verbal content matches tone and body language. Incongruence is the root of most relationship miscommunication. The agent specifically targets incongruence detection.

**Affective Communication Model (Mehrabian, 1971):** 7% of emotional meaning conveyed by words, 38% by vocal tone, 55% by body language (often misquoted as universal; applies specifically to single-word utterances about feelings).

**Social Penetration Theory (Altman & Taylor, 1973):** Relationships develop through increasing self-disclosure. Avoidant attachment styles arrest this process; anxious styles attempt to accelerate it.

---

## Key Research Papers

| # | Title | Authors | Year | Venue | DOI/Link | Key Finding | Relevance |
|---|-------|---------|------|-------|----------|-------------|-----------|
| 1 | What Predicts Divorce? The Relationship Between Marital Processes and Marital Outcomes | Gottman, J.M. | 1994 | Erlbaum (Book) | ISBN 0-8058-1272-4 | Four Horsemen (criticism, contempt, defensiveness, stonewalling) predict divorce with 93% accuracy over 14 years | Core framework for behavior_classifier.py Horsemen detection |
| 2 | Attachment and Loss, Vol. 1: Attachment | Bowlby, J. | 1969 | Basic Books | ISBN 0-465-00543-8 | Attachment behavioral system is a primary motivational system in humans; proximity-seeking activates in distress | Theoretical foundation for attachment pattern detection |
| 3 | Patterns of Attachment: A Psychological Study of the Strange Situation | Ainsworth, M.D.S. et al. | 1978 | Erlbaum | ISBN 0-89859-461-8 | Identified secure, anxious, and avoidant attachment patterns; each with distinct behavioral and physiological markers | Defines the three attachment categories in behavior_classifier.py |
| 4 | The Five Love Languages: The Secret to Love That Lasts | Chapman, G. | 1992 | Northfield Publishing | ISBN 978-0-8024-1270-6 | People express and receive love differently through one of five primary languages; mismatched languages cause disconnection | Five Love Language signal detection module |
| 5 | Emotions Revealed: Recognizing Faces and Feelings to Improve Communication | Ekman, P. | 2003 | Times Books | ISBN 0-8050-7275-6 | Facial action units (FACS coding system) provide objective, cross-cultural measurement of emotion expression | Basis for VisualAnalyzer AU computation using MediaPipe landmarks |
| 6 | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Baevski, A. et al. | 2020 | NeurIPS 2020 | https://arxiv.org/abs/2006.11477 | Self-supervised contrastive learning on raw audio achieves SOTA on SUPERB benchmark (94.3% phone accuracy) | audio_analyzer.py optional 768-dim embedding via facebook/wav2vec2-base |
| 7 | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford, A. et al. | 2022 | ICML 2023 | https://arxiv.org/abs/2212.04356 | Whisper-large-v3 achieves 2.7% WER on LibriSpeech clean; supports word-level timestamps; 99 languages | openai/whisper-large-v3 used for transcription in hf_model_manager.py |
| 8 | More Than a Feeling: Accuracy and Application of Sentiment Analysis | Hartmann, J. | 2022 | IEMBA Dissertation | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base | DistilRoBERTa fine-tuned on 6 datasets achieves F1=0.66 macro on GoEmotions; 7-class emotion classification | j-hartmann/emotion-english-distilroberta-base used in BehaviorClassifier |
| 9 | MediaPipe: A Framework for Building Perception Pipelines | Lugaresi, C. et al. | 2019 | arXiv | https://arxiv.org/abs/1906.08172 | MediaPipe FaceMesh detects 468 3D facial landmarks at 30+ FPS on mobile hardware | visual_analyzer.py landmark extraction and AU computation foundation |
| 10 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin, J. et al. | 2019 | NAACL 2019 | https://arxiv.org/abs/1810.04805 | Bidirectional pre-training with masked language modeling achieves SOTA across 11 NLP benchmarks | Foundation for emotion classification models; BAAI/bge embeddings build on BERT architecture |
| 11 | The Seven Principles for Making Marriage Work | Gottman, J.M. & Silver, N. | 1999 | Crown Publishers | ISBN 0-609-60144-0 | Positive Sentiment Override (PSO) buffers against Four Horsemen impact; repair attempts crucial for recovery | Adds PSO/NSO context to interpretation_engine.py report generation |
| 12 | Attachment in Adulthood: Structure, Dynamics, and Change | Mikulincer, M. & Shaver, P.R. | 2007 | Guilford Press | ISBN 978-1-59385-457-4 | Adult attachment patterns are stable but changeable; secure base scripting influences relationship quality | Informs attachment pattern interpretation in InterpretationEngine |
| 13 | Betrayal, Rejection, Revenge, and Forgiveness: An Interpersonal Script Approach | Fitness, J. | 2001 | Interpersonal Rejection (book) | Mark Leary (Ed.), Oxford UP | Emotion scripts in relationships follow predictable patterns; contextual knowledge shapes emotional interpretation | Provides emotion script patterns for non-literal expression decoder |
| 14 | Marital Interaction: Physiological Linkage and Affective Exchange | Levenson, R.W. & Gottman, J.M. | 1994 | Journal of Personality and Social Psychology | https://doi.org/10.1037/0022-3514.67.3.363 | Physiological linkage (heart rate, skin conductance) correlates with relationship satisfaction; stonewalling preceded by HR>100 | Informs prosodic stress markers as proxy for physiological arousal |
| 15 | Multimodal Sentiment Analysis with Word-Level Fusion and Reinforcement Learning | Kim, J. et al. | 2018 | ICMI 2018 | https://doi.org/10.1145/3242969.3242990 | Late-fusion multimodal sentiment achieves 78.9% accuracy on CMU-MOSI benchmark; word-level alignment improves results | Validates late-fusion architecture chosen for audio+text+visual integration |

---

## State-of-the-Art Models

| Model | Task | Benchmark | Score | Date | Notes |
|-------|------|-----------|-------|------|-------|
| `facebook/wav2vec2-base` | Speech representation | SUPERB avg | 74.1 | 2021-06 | Best self-supervised baseline; no fine-tuning needed for prosodic features |
| `j-hartmann/emotion-english-distilroberta-base` | 7-class emotion (text) | GoEmotions F1 macro | 0.66 | 2022-10 | Lightweight; 7 classes include all relationship-relevant emotions |
| `openai/whisper-large-v3` | ASR (English) | LibriSpeech WER clean | 2.7% | 2023-11 | Best open ASR; word timestamps critical for prosody-text alignment |
| `BAAI/bge-large-en-v1.5` | Text embedding | MTEB avg | 64.23 | 2023-09 | Top open embedding model; outperforms OpenAI text-embedding-ada-002 on most tasks |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | BEIR nDCG@10 | 0.491 | 2023-09 | Best open reranker; critical for RAG quality in knowledge retrieval |
| `facebook/bart-large-cnn` | Summarization | CNN-DM ROUGE-2 | 21.28 | 2020-10 | Reliable abstractive summarization for session transcripts |

---

## LLM Prompt Templates

### Template 1: EMPATHY_REPORT_PROMPT
```
System:
You are a compassionate relationship science expert trained in Gottman Method Couples Therapy,
Ainsworth Attachment Theory, and Chapman's Five Love Languages. You help users understand and
empathize with their partner's emotional state. This is an empathy and self-improvement tool —
NOT a manipulation or surveillance tool.

Analyze the following multimodal communication data and produce a structured empathy report.

MULTIMODAL CONTEXT:
{multimodal_context}

COMMUNICATION FEATURES:
- Gottman Horsemen Scores: {horsemen_scores}
- Attachment Pattern: {attachment_pattern} (confidence: {attachment_confidence})
- Love Language Signals: {love_language_scores}
- Non-Literal Expression Detected: {non_literal_detected}
- Non-Literal Phrase: {non_literal_phrase}
- Actual Emotional State: {actual_emotional_state}
- Overall Distress Score: {distress_score}
- Emotion Distribution (text): {emotion_scores}
- Prosodic Features: F0={f0_mean}Hz (±{f0_std}), speech_rate={speech_rate} syl/s, pauses={pause_count}

SUPPORTING RESEARCH:
{retrieved_papers}

Respond ONLY with valid JSON. Do not add markdown code fences.
{
  "actual_feeling": "what the partner is actually feeling (1-2 sentences)",
  "stated_vs_actual_gap": "explain the incongruence between what was said and what was meant (if any)",
  "gottman_pattern_active": "which Horseman is most active, or 'none'",
  "attachment_context": "what attachment theory says about this pattern (1-2 sentences)",
  "love_language_expressed": "which love language the partner is expressing or seeking",
  "empathy_responses": [
    "Option 1: specific empathy response the user can give",
    "Option 2: alternative empathy response",
    "Option 3: third empathy response"
  ],
  "urgency_level": "low|medium|high|counseling_needed",
  "counseling_recommended": false,
  "counseling_message": "",
  "confidence_explanation": "brief note on what signals were most informative",
  "supporting_research": ["Gottman 1994", "Ainsworth 1978"]
}

SAFETY RULES (MANDATORY):
1. Frame empathy_responses as OPTIONS ("you might try..."), never commands
2. Never suggest specific medication, diagnoses, or harmful actions
3. If counseling_recommended=true, counseling_message must be non-empty and supportive
4. Never frame this as surveillance or manipulation
5. Acknowledge uncertainty — you are analyzing signals, not reading minds
```

### Template 2: HEDGING_ANALYSIS_PROMPT
```
System: You are an expert in pragmatic linguistics, specifically indirect communication and hedging.

Analyze the following text for hedging language, indirect expressions, and non-literal meaning:

TEXT: "{text}"

PROSODIC CONTEXT: F0_variance={f0_variance}, speech_rate={speech_rate}, energy={energy_mean}

Respond with JSON only:
{
  "hedging_detected": boolean,
  "hedging_phrases": ["phrase1", "phrase2"],
  "literal_meaning": "what the words literally say",
  "probable_actual_meaning": "what the speaker probably means",
  "confidence": 0.0-1.0,
  "linguistic_evidence": "brief explanation"
}
```

### Template 3: GOTTMAN_ANALYSIS_PROMPT
```
System: You are trained in the Gottman Method and can identify the Four Horsemen in communication.

Apply Gottman's Four Horsemen framework to the following communication data:
TEXT: "{text}"
TONE SIGNALS: {prosodic_summary}
VISUAL SIGNALS: {visual_summary}

Score each Horseman 0.0–1.0 and provide linguistic evidence:
{
  "criticism": {"score": 0.0, "evidence": ""},
  "contempt": {"score": 0.0, "evidence": ""},
  "defensiveness": {"score": 0.0, "evidence": ""},
  "stonewalling": {"score": 0.0, "evidence": ""},
  "overall_concern": "low|medium|high",
  "primary_horseman": "criticism|contempt|defensiveness|stonewalling|none"
}
```

### Template 4: COUNSELING_RECOMMENDATION_PROMPT
```
System: You are a caring mental health communication specialist.

The communication analysis has detected significant relationship distress patterns (distress_score={score:.2f}).
Active patterns: {active_patterns}

Generate a compassionate, non-alarmist recommendation for professional couples counseling.
Emphasize:
- Seeking support is a sign of strength, not failure
- Professional counselors have tools not available in this application
- This app is an empathy aid, not a replacement for professional care
- Many couples improve dramatically with professional guidance

Keep the message under 150 words. Be warm and encouraging.
```

---

## Authoritative Data Sources

| Source | URL | Purpose | Access |
|--------|-----|---------|--------|
| ArXiv cs.CL | https://export.arxiv.org/api/query?cat=cs.CL | NLP/communication papers | Public API |
| ArXiv cs.CV | https://export.arxiv.org/api/query?cat=cs.CV | Facial expression/vision papers | Public API |
| ArXiv cs.HC | https://export.arxiv.org/api/query?cat=cs.HC | Human-computer interaction | Public API |
| Semantic Scholar Graph API | https://api.semanticscholar.org/graph/v1/paper/search | Academic paper search | Public, rate-limited |
| Gottman Institute | https://www.gottman.com/blog/ | Relationship research blog | Web crawl |
| APA PsycNet | https://psycnet.apa.org | Psychology journal abstracts | Requires subscription (abstract-level free) |
| Journal of Personality and Social Psychology | https://www.apa.org/pubs/journals/psp | Relationship emotion research | Abstract-level public |
| Papers with Code (emotion) | https://paperswithcode.com/task/emotion-recognition | SOTA model leaderboards | Public |

---

## Self-Update Protocol

```yaml
schedule: weekly, Sunday 02:00 local time
sources:
  arxiv:
    categories: [cs.CL, cs.CV, cs.HC]
    queries:
      - "multimodal emotion recognition couple communication"
      - "attachment theory natural language processing"
      - "Gottman Four Horsemen detection machine learning"
      - "speech prosody relationship quality"
      - "facial action unit emotion recognition"
    max_results_per_query: 10
    lookback_days: 90
  semantic_scholar:
    queries:
      - "female partner communication emotion analysis"
      - "attachment theory behavioral markers NLP"
      - "gottman communication patterns machine learning"
      - "multimodal emotion recognition speech text"
      - "facial action unit relationship communication"
    max_results_per_query: 10
scoring:
  recency_weight: 0.6
  relevance_weight: 0.4
  domain_keywords:
    - emotion recognition
    - attachment theory
    - gottman
    - communication patterns
    - facial expression
    - speech prosody
    - relationship quality
    - affective computing
    - sentiment analysis
    - love language
    - multimodal
    - couple communication
    - hedging language
    - non-verbal cues
    - partner behavior
deduplication: SHA256 hash of (URL or DOI)
append_target: SECOND-KNOWLEDGE-BRAIN.md (Key Research Papers table + Knowledge Update Log)
notification: print "N new papers added; next run: {next_sunday}"
```

---

## Knowledge Update Log

| Date | Papers Added | Sources | Top New Finding |
|------|-------------|---------|----------------|
| 2026-06-11 | 15 | Manual curation (initial) | Gottman Four Horsemen predict divorce with 93% accuracy (Gottman 1994); wav2vec2 self-supervised learning achieves SOTA on SUPERB benchmark (Baevski 2020); late-fusion multimodal sentiment achieves 78.9% on CMU-MOSI (Kim 2018) |
