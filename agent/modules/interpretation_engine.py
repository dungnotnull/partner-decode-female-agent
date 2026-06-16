"""
interpretation_engine.py — LLM-powered empathy report synthesis.

Synthesizes multimodal classification results into structured empathy reports
using Claude (primary), OpenAI (fallback), or Ollama (offline) as reasoning engine.
Safety gates: never diagnose, never prescribe, always recommend professional
counseling when high-conflict patterns are detected.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.modules.audio_analyzer import ProsodicFeatures
from agent.modules.behavior_classifier import ClassificationResult
from agent.modules.visual_analyzer import VisualFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

EMPATHY_REPORT_SYSTEM_PROMPT = """You are a compassionate relationship science expert trained in:
- Gottman Method Couples Therapy (Four Horsemen framework)
- Ainsworth Attachment Theory (secure/anxious/avoidant styles)
- Chapman's Five Love Languages
- Pragmatic linguistics (non-literal and hedging language detection)

Your role is to help the user UNDERSTAND and EMPATHIZE with their partner.
This is an empathy and self-improvement tool — NOT a manipulation or surveillance tool.

MANDATORY SAFETY RULES:
1. Frame empathy_responses as OPTIONS ("you might try..."), never commands
2. Never suggest specific medication, clinical diagnoses, or harmful actions
3. If counseling_recommended=true, counseling_message must be non-empty and compassionate
4. Never frame the analysis as surveillance or manipulation
5. Acknowledge uncertainty — you are analyzing signals, not reading minds
6. Always include "supporting_research" citations from the Gottman/Ainsworth/Chapman corpus

Respond ONLY with valid JSON. Do NOT wrap in markdown code fences."""

EMPATHY_REPORT_USER_TEMPLATE = """Analyze the following multimodal communication data and produce a structured empathy report.

MULTIMODAL CONTEXT:
Text spoken: "{text}"
Transcription method: {transcription_method}

AUDIO PROSODIC FEATURES:
- F0 (pitch): mean={f0_mean:.1f}Hz, std={f0_std:.1f}Hz, range={f0_range:.1f}Hz
- Speech rate: {speech_rate:.1f} syllables/sec
- Pauses: count={pause_count}, ratio={pause_ratio:.2f}
- Energy: mean={energy_mean:.4f}, std={energy_std:.4f}
- RMS: {rms:.4f}
- Vocal cues: {vocal_cues}
- Duration: {duration:.1f}s

BEHAVIOR CLASSIFICATION RESULTS:
- Gottman Horsemen: criticism={criticism:.2f}, contempt={contempt:.2f}, defensiveness={defensiveness:.2f}, stonewalling={stonewalling:.2f}
- Attachment Pattern: {attachment_pattern} (confidence={attachment_confidence:.2f})
- Love Language Signals: {love_language_signals}
- Non-Literal Expression Detected: {non_literal_detected}
- Non-Literal Phrase: "{non_literal_phrase}"
- Actual Emotional State: {actual_emotional_state}
- Communication Style: {communication_style}
- Overall Distress Score: {distress_score:.2f}

EMOTION SCORES (text-based):
{emotion_scores}

VISUAL FEATURES:
{visual_summary}

SUPPORTING RESEARCH CONTEXT:
{retrieved_papers}

Produce the following JSON structure:
{{
  "actual_feeling": "what the partner is actually feeling (1-2 sentences, specific and compassionate)",
  "stated_vs_actual_gap": "explain the gap between what was said and what was meant (or 'No significant gap detected')",
  "gottman_pattern_active": "most active Horseman name and brief explanation, or 'None — communication appears constructive'",
  "attachment_context": "brief Attachment Theory explanation of this communication pattern (1-2 sentences)",
  "love_language_expressed": "which love language the partner is expressing OR seeking",
  "empathy_responses": [
    "Option 1: a specific empathy response phrased as 'You might try...'",
    "Option 2: an alternative empathy response",
    "Option 3: a third empathy response if the first two feel difficult"
  ],
  "urgency_level": "low|medium|high|counseling_needed",
  "counseling_recommended": false,
  "counseling_message": "",
  "confidence_explanation": "brief note on which signals were most informative and any limitations",
  "supporting_research": ["citation1 (Author Year)", "citation2 (Author Year)"]
}}"""

HEDGING_ANALYSIS_PROMPT = """Analyze the following text for hedging language and indirect communication patterns.

TEXT: "{text}"
PROSODIC CONTEXT: F0_variance={f0_variance:.1f}Hz, speech_rate={speech_rate:.1f} syl/s, energy={energy_mean:.4f}

Respond with JSON only:
{{
  "hedging_detected": true/false,
  "hedging_phrases": ["phrase1", "phrase2"],
  "literal_meaning": "what the words literally say",
  "probable_actual_meaning": "what the speaker probably means",
  "confidence": 0.0,
  "linguistic_evidence": "brief explanation of hedging indicators"
}}"""

SAFETY_GATE_COUNSELING_PROMPT = """The communication analysis has detected significant relationship distress.

Distress Score: {distress_score:.2f}
Active Patterns: {active_patterns}

Generate a compassionate, non-alarmist recommendation for professional couples counseling.
Emphasize:
- Seeking support is a sign of strength and love
- Professional counselors have specialized tools this app cannot provide
- This app is an empathy aid, not a replacement for professional care
- Many couples experience significant improvement with professional guidance
- Framing: "considering speaking with a couples therapist" rather than "you must see a therapist"

Write under 150 words. Use warm, encouraging language. Do not alarm. Return plain text (no JSON)."""

# ---------------------------------------------------------------------------
# Fallback interpretations when LLM is unavailable
# ---------------------------------------------------------------------------

FALLBACK_INTERPRETATIONS: Dict[str, dict] = {
    "direct": {
        "actual_feeling": "Your partner appears to be communicating directly. The emotional signals are relatively clear.",
        "stated_vs_actual_gap": "No significant gap detected between stated content and emotional signals.",
        "gottman_pattern_active": "None detected — communication appears constructive.",
        "attachment_context": "The communication pattern suggests a relatively secure attachment base.",
        "love_language_expressed": "Unable to determine dominant love language from available signals.",
        "empathy_responses": [
            "You might try acknowledging what you heard: 'I hear that you're feeling...'",
            "You might try asking an open question: 'Can you tell me more about that?'",
            "You might try validating their experience: 'That makes sense that you would feel that way.'",
        ],
        "urgency_level": "low",
        "counseling_recommended": False,
        "counseling_message": "",
        "confidence_explanation": "Analysis based on text patterns only (LLM unavailable — offline mode).",
        "supporting_research": ["Gottman 1994", "Ainsworth 1978"],
    },
    "indirect": {
        "actual_feeling": "Your partner may be expressing something different from what the literal words convey. This is a common indirect communication pattern.",
        "stated_vs_actual_gap": "Possible gap detected between stated words and underlying emotional need.",
        "gottman_pattern_active": "Possible defensiveness or withdrawal patterns in communication style.",
        "attachment_context": "Indirect communication is often associated with anxious or avoidant attachment — difficulty expressing needs directly.",
        "love_language_expressed": "The indirect expression may indicate a need for Words of Affirmation or Quality Time.",
        "empathy_responses": [
            "You might try gently naming what you notice: 'I notice there might be more to this — I want to hear it if you're ready to share.'",
            "You might try creating safety: 'There's nothing you could tell me that I couldn't hear. I'm here.'",
            "You might try asking directly: 'Are you actually okay, or is there something you need from me right now?'",
        ],
        "urgency_level": "medium",
        "counseling_recommended": False,
        "counseling_message": "",
        "confidence_explanation": "Indirect communication pattern detected via text analysis (LLM unavailable — offline mode).",
        "supporting_research": ["Gottman 1994 (indirect expression)", "Chapman 1992 (love languages)"],
    },
    "hedging": {
        "actual_feeling": "Your partner appears uncertain or uncomfortable expressing their full feelings directly.",
        "stated_vs_actual_gap": "Hedging language suggests the stated words may not fully represent the emotional truth.",
        "gottman_pattern_active": "Possible stonewalling or defensive hedging.",
        "attachment_context": "Hedging is a common communication pattern in anxious attachment — feeling unsafe to express needs clearly.",
        "love_language_expressed": "Hedging may indicate unmet need for Words of Affirmation or Quality Time.",
        "empathy_responses": [
            "You might try removing pressure: 'Take your time. I'm not going anywhere.'",
            "You might try reflecting: 'It sounds like there's something you're working through. I want to understand.'",
            "You might try open-ended invitation: 'Whatever it is, I'd like to hear it when you're ready.'",
        ],
        "urgency_level": "medium",
        "counseling_recommended": False,
        "counseling_message": "",
        "confidence_explanation": "Hedging language pattern detected (LLM unavailable — offline mode).",
        "supporting_research": ["Ainsworth 1978", "Gottman 1994"],
    },
    "withdrawn": {
        "actual_feeling": "Your partner appears to be emotionally withdrawn. This can be a sign of emotional overwhelm or the stonewalling pattern.",
        "stated_vs_actual_gap": "Withdrawn communication often hides intense underlying emotion.",
        "gottman_pattern_active": "Stonewalling detected — emotional withdrawal from interaction.",
        "attachment_context": "Stonewalling often precedes or follows emotional flooding (heart rate > 100 BPM). It is a self-protective response, not always a hostile one.",
        "love_language_expressed": "Withdrawal may indicate an unmet need for Quality Time or Words of Affirmation.",
        "empathy_responses": [
            "You might try offering a break: 'I can see this is a lot right now. We can continue when you're ready — I'm not going anywhere.'",
            "You might try reducing pressure: 'You don't have to respond immediately. I just want you to know I care about how you're feeling.'",
            "You might try physical presence without demands: being nearby without requiring conversation can help someone who is flooded.",
        ],
        "urgency_level": "high",
        "counseling_recommended": False,
        "counseling_message": "",
        "confidence_explanation": "Stonewalling/withdrawal pattern detected (LLM unavailable — offline mode).",
        "supporting_research": ["Gottman 1994 (stonewalling)", "Levenson & Gottman 1994 (physiological flooding)"],
    },
}

# Prohibited output patterns (harm prevention scan)
PROHIBITED_PATTERNS = [
    r"\btake (this |the )?(medication|drug|pill|antidepressant|anti-anxiety)\b",
    r"\bdiagnose[sd]? with\b",
    r"\bhave (borderline|narcissistic|bipolar|schizophrenic)\b",
    r"\bmanipulate (her|them|your partner)\b",
    r"\bspy on\b",
    r"\bgaslighting (her|them|your partner)\b",
    r"\bcontrol (her|them|your partner)\b",
]


@dataclass
class InterpretationResult:
    """Container for the LLM-synthesized empathy report."""

    actual_feeling: str = ""
    stated_vs_actual_gap: str = ""
    gottman_pattern_active: str = ""
    attachment_context: str = ""
    love_language_expressed: str = ""
    empathy_responses: List[str] = field(default_factory=list)
    urgency_level: str = "low"
    counseling_recommended: bool = False
    counseling_message: str = ""
    confidence_explanation: str = ""
    supporting_research: List[str] = field(default_factory=list)

    # Metadata
    llm_provider_used: str = ""
    fallback_used: bool = False
    raw_llm_response: str = ""


class InterpretationEngine:
    """
    Synthesizes multimodal classification into natural language empathy reports
    using LLM API (Claude → OpenAI → Ollama).

    Includes harm prevention scan on all outputs.
    """

    def __init__(
        self,
        llm_client=None,
        hf_model_manager=None,
        knowledge_brain_path: str = "./SECOND-KNOWLEDGE-BRAIN.md",
    ) -> None:
        self._llm = llm_client
        self._hf = hf_model_manager
        self._knowledge_brain_path = knowledge_brain_path
        self._compiled_prohibited = [
            re.compile(p, re.IGNORECASE) for p in PROHIBITED_PATTERNS
        ]

    def interpret(
        self,
        classification: ClassificationResult,
        audio_features: ProsodicFeatures,
        text: str,
        visual_features: Optional[VisualFeatures] = None,
    ) -> InterpretationResult:
        """
        Main interpretation pipeline:
        1. Build context-rich LLM prompt
        2. Call LLM API
        3. Parse structured JSON response
        4. Apply harm prevention scan
        5. Return InterpretationResult
        """
        # Build the prompt
        prompt = self._build_empathy_prompt(classification, audio_features, text, visual_features)

        # If LLM unavailable, fall back immediately
        if self._llm is None:
            logger.warning("No LLM client configured — using fallback interpretations")
            return self._fallback_result(classification, fallback_reason="no_llm_client")

        # Call LLM
        try:
            response_text, provider_used = self._llm.complete(
                user_prompt=prompt,
                system_prompt=EMPATHY_REPORT_SYSTEM_PROMPT,
                max_tokens=1024,
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("LLM call failed: %s — using fallback", exc)
            return self._fallback_result(classification, fallback_reason=str(exc))

        # Parse response
        parsed = self._parse_llm_response(response_text)
        if parsed is None:
            logger.warning("LLM response parsing failed — using fallback")
            return self._fallback_result(classification, fallback_reason="parse_error")

        # Build result
        result = InterpretationResult(
            actual_feeling=parsed.get("actual_feeling", ""),
            stated_vs_actual_gap=parsed.get("stated_vs_actual_gap", ""),
            gottman_pattern_active=parsed.get("gottman_pattern_active", ""),
            attachment_context=parsed.get("attachment_context", ""),
            love_language_expressed=parsed.get("love_language_expressed", ""),
            empathy_responses=parsed.get("empathy_responses", [])[:3],
            urgency_level=parsed.get("urgency_level", "low"),
            counseling_recommended=bool(parsed.get("counseling_recommended", False)),
            counseling_message=parsed.get("counseling_message", ""),
            confidence_explanation=parsed.get("confidence_explanation", ""),
            supporting_research=parsed.get("supporting_research", []),
            llm_provider_used=provider_used,
            fallback_used=False,
            raw_llm_response=response_text,
        )

        # Safety gate: override counseling_recommended if classification demands it
        if classification.needs_counseling_flag and not result.counseling_recommended:
            result.counseling_recommended = True
            result.urgency_level = "counseling_needed"
            if not result.counseling_message:
                result.counseling_message = self._generate_counseling_message(classification)

        # Harm prevention scan
        result = self._scan_for_harm(result)

        return result

    def format_result(self, result: InterpretationResult) -> str:
        """Format InterpretationResult as a readable ASCII-box report."""
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║          PARTNER DECODE — EMPATHY REPORT                    ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║ Urgency Level: {result.urgency_level.upper():<46}║",
            "╠══════════════════════════════════════════════════════════════╣",
            "║ WHAT SHE'S ACTUALLY FEELING:                                ║",
        ]
        for line in self._wrap_text(result.actual_feeling, 60):
            lines.append(f"║  {line:<60}║")

        if result.stated_vs_actual_gap and "No significant" not in result.stated_vs_actual_gap:
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ WHAT SHE SAID VS. WHAT SHE MEANT:                           ║")
            for line in self._wrap_text(result.stated_vs_actual_gap, 60):
                lines.append(f"║  {line:<60}║")

        if result.gottman_pattern_active and "None" not in result.gottman_pattern_active:
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ COMMUNICATION PATTERN (Gottman Framework):                  ║")
            for line in self._wrap_text(result.gottman_pattern_active, 60):
                lines.append(f"║  {line:<60}║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append("║ HOW YOU MIGHT RESPOND (3 OPTIONS):                          ║")
        for i, response in enumerate(result.empathy_responses[:3], 1):
            lines.append(f"║  {i}. {'':<58}║")
            for line in self._wrap_text(response, 56):
                lines.append(f"║     {line:<57}║")

        if result.love_language_expressed:
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append(f"║ Love Language Active: {result.love_language_expressed[:40]:<40}║")

        if result.attachment_context:
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ Attachment Theory Context:                                  ║")
            for line in self._wrap_text(result.attachment_context, 60):
                lines.append(f"║  {line:<60}║")

        if result.counseling_recommended and result.counseling_message:
            lines.append("╠══════════════════════════════════════════════════════════════╣")
            lines.append("║ *** PROFESSIONAL SUPPORT RECOMMENDATION ***                 ║")
            for line in self._wrap_text(result.counseling_message, 60):
                lines.append(f"║  {line:<60}║")

        lines.append("╠══════════════════════════════════════════════════════════════╣")
        lines.append(f"║ Provider: {result.llm_provider_used:<20} | Fallback: {'Yes' if result.fallback_used else 'No':<6}          ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_empathy_prompt(
        self,
        classification: ClassificationResult,
        audio: ProsodicFeatures,
        text: str,
        visual: Optional[VisualFeatures],
    ) -> str:
        """Assemble the full context prompt for LLM."""
        visual_summary = "No visual data available"
        if visual and visual.available and visual.landmark_count > 0:
            au_highlights = {k: f"{v:.2f}" for k, v in visual.au_scores.items() if v > 0.2}
            visual_summary = (
                f"Expression: {visual.expression_label} (confidence={visual.confidence:.2f}), "
                f"Notable AUs: {au_highlights}, "
                f"Microexpression: {'detected — ' + visual.microexpression_au if visual.microexpression_detected else 'none'}"
            )

        emotion_summary = "\n".join(
            f"  {k}: {v:.2f}" for k, v in sorted(
                classification.emotion_scores.items(), key=lambda x: -x[1]
            )
        ) if classification.emotion_scores else "  Not available"

        retrieved_papers = self._retrieve_relevant_papers(classification)

        love_lang_str = ", ".join(
            f"{k.replace('_', ' ')}={v:.2f}"
            for k, v in classification.love_language_signals.items()
            if v > 0.0
        )

        return EMPATHY_REPORT_USER_TEMPLATE.format(
            text=text or "(no text provided)",
            transcription_method="Whisper ASR" if text else "none",
            f0_mean=audio.f0_mean,
            f0_std=audio.f0_std,
            f0_range=audio.f0_range,
            speech_rate=audio.speech_rate,
            pause_count=audio.pause_count,
            pause_ratio=audio.pause_ratio,
            energy_mean=audio.energy_mean,
            energy_std=audio.energy_std,
            rms=audio.rms,
            vocal_cues=str(audio.vocal_cues),
            duration=audio.duration_seconds,
            criticism=classification.gottman_horsemen.get("criticism", 0.0),
            contempt=classification.gottman_horsemen.get("contempt", 0.0),
            defensiveness=classification.gottman_horsemen.get("defensiveness", 0.0),
            stonewalling=classification.gottman_horsemen.get("stonewalling", 0.0),
            attachment_pattern=classification.attachment_pattern,
            attachment_confidence=classification.attachment_confidence,
            love_language_signals=love_lang_str or "none detected",
            non_literal_detected=classification.non_literal_detected,
            non_literal_phrase=classification.non_literal_phrase,
            actual_emotional_state=classification.actual_emotional_state,
            communication_style=classification.communication_style,
            distress_score=classification.overall_distress_score,
            emotion_scores=emotion_summary,
            visual_summary=visual_summary,
            retrieved_papers=retrieved_papers,
        )

    def _retrieve_relevant_papers(self, classification: ClassificationResult) -> str:
        """Retrieve most relevant research snippets for LLM context."""
        # Simplified retrieval: always include core citations
        citations = [
            "Gottman 1994: Four Horsemen predict divorce with 93% accuracy",
            "Ainsworth 1978: Secure/anxious/avoidant attachment patterns from Strange Situation study",
            "Chapman 1992: Five Love Languages — words of affirmation, acts of service, gifts, quality time, physical touch",
        ]
        if classification.gottman_horsemen.get("contempt", 0) > 0.3:
            citations.append("Gottman 1994: Contempt is the single strongest predictor of divorce")
        if classification.attachment_pattern == "anxious":
            citations.append("Mikulincer & Shaver 2007: Anxious attachment involves hyperactivation of the attachment system")
        if classification.non_literal_detected:
            citations.append("Satir 1972: Congruence theory — incongruence between content and tone is the root of miscommunication")
        return "\n".join(f"- {c}" for c in citations)

    def _parse_llm_response(self, response_text: str) -> Optional[dict]:
        """Parse LLM response as JSON, stripping markdown fences if present."""
        try:
            # Strip markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON object with regex
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    def _fallback_result(
        self, classification: ClassificationResult, fallback_reason: str = ""
    ) -> InterpretationResult:
        """Return a pre-computed fallback interpretation based on communication style."""
        style = classification.communication_style
        fallback_data = FALLBACK_INTERPRETATIONS.get(
            style, FALLBACK_INTERPRETATIONS["direct"]
        )
        result = InterpretationResult(
            **{k: v for k, v in fallback_data.items()},
            llm_provider_used="fallback",
            fallback_used=True,
            raw_llm_response=f"FALLBACK: {fallback_reason}",
        )
        # Apply counseling flag from classification
        if classification.needs_counseling_flag:
            result.counseling_recommended = True
            result.urgency_level = "counseling_needed"
            result.counseling_message = self._generate_counseling_message(classification)
        return result

    def _generate_counseling_message(self, classification: ClassificationResult) -> str:
        """Generate a compassionate counseling recommendation message."""
        active_patterns = []
        for horseman, score in classification.gottman_horsemen.items():
            if score > 0.5:
                active_patterns.append(horseman)
        if classification.attachment_pattern == "anxious":
            active_patterns.append("anxious attachment")

        patterns_str = ", ".join(active_patterns) if active_patterns else "high distress signals"

        return (
            f"The analysis detected {patterns_str} patterns that can be challenging to "
            "navigate alone. Reaching out to a couples therapist or relationship counselor "
            "is a sign of care, not failure. Many couples experience significant breakthroughs "
            "with professional support. Consider exploring the Gottman Referral Network "
            "(gottman.com) for research-backed therapists near you."
        )

    def _scan_for_harm(self, result: InterpretationResult) -> InterpretationResult:
        """Scan all text fields for prohibited content and sanitize if found."""
        text_fields = [
            result.actual_feeling,
            result.stated_vs_actual_gap,
            result.gottman_pattern_active,
            result.attachment_context,
            result.counseling_message,
        ] + result.empathy_responses

        for pattern in self._compiled_prohibited:
            for field_text in text_fields:
                if pattern.search(field_text):
                    logger.warning(
                        "Prohibited content pattern found in LLM output: '%s'. "
                        "Switching to fallback.",
                        pattern.pattern[:50],
                    )
                    # Replace entire result with safe fallback
                    result.actual_feeling = (
                        "This analysis detected content that could not be safely presented. "
                        "Please consult a relationship professional for guidance."
                    )
                    result.empathy_responses = [
                        "You might try listening actively without judgment.",
                        "You might try asking how you can support them right now.",
                        "You might try acknowledging that you hear them and that you care.",
                    ]
                    result.counseling_recommended = True
                    result.counseling_message = self._generate_counseling_message(
                        ClassificationResult()
                    )
                    result.fallback_used = True
                    return result
        return result

    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to fit in ASCII box display."""
        if not text:
            return [""]
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current += (" " if current else "") + word
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]
