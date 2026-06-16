"""
behavior_classifier.py — Multimodal behavior pattern classification.

Applies Gottman's Four Horsemen framework, Ainsworth Attachment Theory,
Chapman's Five Love Languages, and non-literal expression detection to
prosodic + text + visual features.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent.modules.audio_analyzer import ProsodicFeatures
from agent.modules.visual_analyzer import VisualFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gottman Four Horsemen Pattern Definitions
# ---------------------------------------------------------------------------

CRITICISM_PATTERNS: List[str] = [
    r"\byou always\b",
    r"\byou never\b",
    r"\byou are so\b",
    r"\byou're so\b",
    r"\bwhat is wrong with you\b",
    r"\byou can't\b.*\banything\b",
    r"\byou don't\b.*\never\b",
    r"\bjust like\b.*\balways\b",
    r"\bwhy do you always\b",
    r"\bevery time\b.*\byou\b",
    r"\byou're the (kind|type) of person\b",
    r"\byou just don't\b",
    r"\byou never listen\b",
    r"\btypical\b.*\byou\b",
    r"\byou have no\b.{0,30}(sense|respect|idea|clue)",
    r"\bnobody else\b.*\byou\b",
    r"\byou're impossible\b",
    r"\byou're hopeless\b",
    r"\bwhat's wrong with you\b",
    r"\bof course you\b",
]

CONTEMPT_PATTERNS: List[str] = [
    r"\bthat's ridiculous\b",
    r"\bhow pathetic\b",
    r"\byou're so naive\b",
    r"\bwhat a joke\b",
    r"\byou don't even know\b",
    r"\bplease\b.*\byou think\b",
    r"\byeah right\b",
    r"\bwhatever you say\b",
    r"\bif you say so\b",
    r"\bsure, sure\b",
    r"\boh wow\b.*\bincredible\b",
    r"\bthat figures\b",
    r"\bof course\b.*\bwould\b.*\bthink\b",
    r"\byou actually believe\b",
    r"\byou must be kidding\b",
]

DEFENSIVENESS_PATTERNS: List[str] = [
    r"\bbut i\b",
    r"\bit's not my fault\b",
    r"\bi didn't do anything\b",
    r"\byes, but\b",
    r"\bthat's not fair\b",
    r"\byou're wrong\b",
    r"\byou don't understand\b",
    r"\bi was just\b",
    r"\bi never said\b",
    r"\bdon't blame me\b",
    r"\bwhy are you always\b.{0,30}\bme\b",
    r"\byou started\b",
    r"\bwhat about you\b",
    r"\bi tried\b",
    r"\bnobody tells me\b",
]

STONEWALLING_PATTERNS: List[str] = [
    r"^\s*(ok\.?|fine\.?|whatever\.?|sure\.?|mmm\.?|uh-huh\.?|yeah\.?)\s*$",
    r"^\s*\.\.\.\s*$",
    r"^\s*(i don't want to talk about this)\s*$",
    r"^\s*(not now)\s*$",
    r"^\s*(drop it)\s*$",
    r"^\s*(leave me alone)\s*$",
    r"^\s*(i'm done)\s*$",
    r"^\s*(forget it)\s*$",
]

# ---------------------------------------------------------------------------
# Attachment Pattern Definitions
# ---------------------------------------------------------------------------

ANXIOUS_PATTERNS: List[str] = [
    r"\bdo you (still )?love me\b",
    r"\bare you sure\b",
    r"\bdo you (still )?care\b",
    r"\byou don't care\b",
    r"\byou're going to leave\b",
    r"\bwhat did i do wrong\b",
    r"\bplease don't (be mad|ignore me|leave)\b",
    r"\bi need you\b",
    r"\bwhy aren't you answering\b",
    r"\byou always leave me\b",
    r"\bi'm scared\b.*\b(lose|losing) you\b",
    r"\bam i\b.*\benough\b",
    r"\byou'll (leave|forget|abandon)\b",
]

AVOIDANT_PATTERNS: List[str] = [
    r"\bi need space\b",
    r"\bi'm fine\b",
    r"\bi don't want to talk about it\b",
    r"\bit doesn't matter\b",
    r"\bwhatever\b",
    r"\bi don't care\b",
    r"\bjust drop it\b",
    r"\bi'm not (mad|upset|angry)\b",
    r"\bstop making a big deal\b",
    r"\byou're overreacting\b",
    r"\bi don't have feelings about this\b",
    r"\bcan we just\b.*\bmove on\b",
]

SECURE_PATTERNS: List[str] = [
    r"\bi understand\b",
    r"\bi hear you\b",
    r"\bthat makes sense\b",
    r"\bi appreciate\b",
    r"\bthank you for telling me\b",
    r"\bi love you and\b",
    r"\bwe can figure this out\b",
    r"\bi'm here for you\b",
    r"\blet's talk about this\b",
    r"\bi was wrong\b",
    r"\bi'm sorry\b",
    r"\bwe're okay\b",
]

# ---------------------------------------------------------------------------
# Five Love Language Signal Patterns
# ---------------------------------------------------------------------------

LOVE_LANGUAGE_PATTERNS: Dict[str, List[str]] = {
    "words_of_affirmation": [
        r"\bi love you\b",
        r"\byou're amazing\b",
        r"\byou did (a great|an amazing|a good) job\b",
        r"\bthank you for\b",
        r"\bi appreciate\b.*\byou\b",
        r"\byou're so (kind|wonderful|great|beautiful)\b",
        r"\bi'm proud of you\b",
        r"\byou mean so much\b",
        r"\byou're (my )?everything\b",
        r"\bi'm grateful for you\b",
    ],
    "acts_of_service": [
        r"\bcould you (please )?(help|do)\b",
        r"\bwould you\b.*\bfor me\b",
        r"\bcan you\b.*\btoday\b",
        r"\bi need help with\b",
        r"\byou (always )?take care of\b",
        r"\bthanks for (fixing|making|doing|cooking|cleaning)\b",
        r"\bi wish you (would|could) (do|help)\b",
    ],
    "receiving_gifts": [
        r"\bi (saw|found) this and thought of you\b",
        r"\bi got you\b",
        r"\bi picked (this|it) up for you\b",
        r"\bdo you like your (gift|present|surprise)\b",
        r"\bi remembered you said you wanted\b",
        r"\bit reminded me of you\b",
        r"\bsmall (gift|surprise|token)\b",
    ],
    "quality_time": [
        r"\bwe should\b.*\btogether\b",
        r"\bi miss (us|spending time with you)\b",
        r"\bcan we\b.*\bboth\b",
        r"\blet's (do something|go|have|spend time)\b.*\btogether\b",
        r"\bi want to\b.*\bwith you\b",
        r"\bwe haven't\b.*\bin so long\b",
        r"\bjust the two of us\b",
        r"\byou have time for (me|us)\b",
    ],
    "physical_touch": [
        r"\bcan (i|we)\b.*\b(hug|hold|touch|cuddle|kiss)\b",
        r"\bhold (my hand|me)\b",
        r"\bi need a hug\b",
        r"\bcome (sit|be) (close|near)\b",
        r"\bi (miss|want) (your touch|being close|your arms)\b",
        r"\bphysical(ly)?\b.*\bclose\b",
        r"\bbody language\b",
    ],
}

# ---------------------------------------------------------------------------
# Non-Literal Expression Dictionary
# ---------------------------------------------------------------------------

# Format: (text_pattern, expected_prosodic_cue, actual_state, likely_meaning)
INCONGRUENCE_PHRASES: List[Tuple[str, str, str, str]] = [
    (r"i'm fine", "stressed_prosody", "stressed", "I am not fine and need acknowledgment"),
    (r"it's fine", "high_energy", "upset", "It is not fine; I want you to notice"),
    (r"whatever", "rising_f0", "frustrated", "I feel dismissed and am withdrawing"),
    (r"do whatever you want", "flat_affect", "disengaged", "I feel my preferences are not valued"),
    (r"forget it", "high_energy", "hurt", "I want you to pursue the conversation"),
    (r"i don't care", "stressed_prosody", "cares_a_lot", "I care deeply but feel unable to say so"),
    (r"i'm not (mad|upset|angry)", "high_f0_variance", "angry", "I am angry but suppressing it"),
    (r"no, nothing", "elevated_energy", "something_is_wrong", "Something is wrong; I want to be asked"),
    (r"sure", "flat_affect", "reluctant", "I am not sure; I am agreeing to avoid conflict"),
    (r"ok", "minimal_energy", "resigned", "I am giving up on being understood"),
    (r"it doesn't matter", "stressed_prosody", "matters_greatly", "It matters very much to me"),
    (r"i guess", "rising_f0", "uncertain_seeking_validation", "I need reassurance about this"),
    (r"never mind", "high_energy", "frustrated", "I want you to ask what is wrong"),
    (r"i'm not bothered", "fast_speech_rate", "bothered", "I am bothered but masking it"),
    (r"don't worry about it", "clipped_tone", "worried_myself", "I am worried but withdrawing"),
    (r"i'll be fine", "low_energy", "struggling", "I need support but am afraid to ask"),
    (r"just go", "high_f0_variance", "wants_you_to_stay", "I want you to choose to stay"),
    (r"it's nothing", "stressed_prosody", "significant_concern", "It is not nothing; please ask"),
    (r"i'm not even (mad|upset|hurt)", "flat_affect", "deeply_hurt", "I am hurt and shutting down"),
    (r"you know what, forget it", "clipped_tone", "needs_resolution", "I need this resolved but feel unheard"),
]


@dataclass
class ClassificationResult:
    """Container for all behavior classification results."""

    # Gottman Four Horsemen scores (0.0–1.0 each)
    gottman_horsemen: Dict[str, float] = field(default_factory=lambda: {
        "criticism": 0.0,
        "contempt": 0.0,
        "defensiveness": 0.0,
        "stonewalling": 0.0,
    })

    # Attachment Pattern
    attachment_pattern: str = "secure"   # anxious / avoidant / secure
    attachment_confidence: float = 0.0

    # Five Love Language signals (0.0–1.0 each)
    love_language_signals: Dict[str, float] = field(default_factory=lambda: {
        "words_of_affirmation": 0.0,
        "acts_of_service": 0.0,
        "receiving_gifts": 0.0,
        "quality_time": 0.0,
        "physical_touch": 0.0,
    })

    # Non-literal expression
    non_literal_detected: bool = False
    non_literal_phrase: str = ""
    actual_emotional_state: str = ""
    probable_meaning: str = ""

    # Emotion scores from text classifier (7 classes)
    emotion_scores: Dict[str, float] = field(default_factory=dict)

    # Aggregate metrics
    communication_style: str = "direct"   # direct / indirect / hedging / withdrawn
    overall_distress_score: float = 0.0
    needs_counseling_flag: bool = False
    confidence: float = 0.0


class BehaviorClassifier:
    """
    Applies validated relationship-science frameworks to multimodal features.

    Frameworks:
    - Gottman's Four Horsemen (Gottman 1994)
    - Attachment Theory — Ainsworth 3-style model (Ainsworth 1978)
    - Five Love Languages (Chapman 1992)
    - Non-literal expression detection (Satir 1972; pragmatic linguistics)

    Safety gate: if distress_score > 0.8 or any Horseman > 0.7 → needs_counseling_flag=True
    """

    # Safety thresholds (hard-coded, not configurable — see PROJECT-detail.md)
    COUNSELING_DISTRESS_THRESHOLD = 0.8
    COUNSELING_HORSEMAN_THRESHOLD = 0.7

    def __init__(self, hf_model_manager=None) -> None:
        self._hf_manager = hf_model_manager
        self._compile_patterns()

    def classify(
        self,
        audio_features: ProsodicFeatures,
        text: str,
        visual_features: Optional[VisualFeatures] = None,
    ) -> ClassificationResult:
        """
        Main classification pipeline: apply all frameworks to multimodal features.
        """
        result = ClassificationResult()
        text_lower = text.lower().strip() if text else ""

        # --- Step 1: Emotion classification via HF model (if available) ---
        if self._hf_manager is not None and text_lower:
            try:
                result.emotion_scores = self._hf_manager.classify_emotion(text_lower)
            except Exception as exc:
                logger.debug("Emotion classification failed: %s", exc)

        # --- Step 2: Four Horsemen scoring ---
        result.gottman_horsemen = self._score_gottman_horsemen(
            text_lower, audio_features, visual_features
        )

        # --- Step 3: Attachment pattern detection ---
        result.attachment_pattern, result.attachment_confidence = (
            self._detect_attachment_pattern(audio_features, text_lower)
        )

        # --- Step 4: Love language signal detection ---
        result.love_language_signals = self._detect_love_languages(text_lower)

        # --- Step 5: Non-literal expression detection ---
        (
            result.non_literal_detected,
            result.non_literal_phrase,
            result.actual_emotional_state,
            result.probable_meaning,
        ) = self._detect_non_literal(text_lower, audio_features)

        # --- Step 6: Communication style ---
        result.communication_style = self._infer_communication_style(
            result, text_lower, audio_features
        )

        # --- Step 7: Distress score ---
        result.overall_distress_score = self._compute_distress_score(
            result.gottman_horsemen,
            result.attachment_pattern,
            audio_features,
            visual_features,
            result.emotion_scores,
        )

        # --- Step 8: Safety gate ---
        max_horseman = max(result.gottman_horsemen.values()) if result.gottman_horsemen else 0.0
        result.needs_counseling_flag = (
            result.overall_distress_score > self.COUNSELING_DISTRESS_THRESHOLD
            or max_horseman > self.COUNSELING_HORSEMAN_THRESHOLD
        )

        # Overall confidence: inversely proportional to feature sparsity
        has_audio = audio_features.rms > 0.001
        has_text = len(text_lower) > 3
        has_visual = visual_features is not None and visual_features.available
        modality_count = sum([has_audio, has_text, has_visual])
        result.confidence = float(modality_count / 3.0)

        return result

    # ------------------------------------------------------------------
    # Private scoring methods
    # ------------------------------------------------------------------

    def _score_gottman_horsemen(
        self,
        text: str,
        audio: ProsodicFeatures,
        visual: Optional[VisualFeatures],
    ) -> Dict[str, float]:
        """Score each of the Four Horsemen using text patterns + prosodic cues."""
        scores = {
            "criticism": 0.0,
            "contempt": 0.0,
            "defensiveness": 0.0,
            "stonewalling": 0.0,
        }

        # --- Criticism ---
        criticism_matches = sum(
            1 for p in self._criticism_compiled
            if p.search(text)
        )
        scores["criticism"] = min(1.0, criticism_matches * 0.25)

        # --- Contempt ---
        contempt_matches = sum(
            1 for p in self._contempt_compiled
            if p.search(text)
        )
        # Contempt boost from visual: AU7+AU9 microexpression or expression_label=contempt
        contempt_visual_boost = 0.0
        if visual and visual.available:
            if visual.expression_label == "contempt":
                contempt_visual_boost = 0.3
            elif visual.microexpression_detected and "AU4" in visual.microexpression_au:
                contempt_visual_boost = 0.15
        scores["contempt"] = min(1.0, contempt_matches * 0.3 + contempt_visual_boost)

        # --- Defensiveness ---
        defensiveness_matches = sum(
            1 for p in self._defensiveness_compiled
            if p.search(text)
        )
        scores["defensiveness"] = min(1.0, defensiveness_matches * 0.25)

        # --- Stonewalling ---
        stonewalling_text = sum(
            1 for p in self._stonewalling_compiled
            if p.search(text)
        )
        # Stonewalling boost from very low audio energy + short text
        stonewall_audio_boost = 0.0
        if audio.rms < 0.005 and audio.duration_seconds > 1.0:
            stonewall_audio_boost = 0.25
        if len(text.split()) <= 3 and audio.rms > 0:
            stonewall_audio_boost += 0.15
        scores["stonewalling"] = min(1.0, stonewalling_text * 0.5 + stonewall_audio_boost)

        return scores

    def _detect_attachment_pattern(
        self,
        audio: ProsodicFeatures,
        text: str,
    ) -> Tuple[str, float]:
        """Classify attachment style as anxious / avoidant / secure with confidence."""
        anxious_score = float(
            sum(1 for p in self._anxious_compiled if p.search(text)) * 0.2
        )
        avoidant_score = float(
            sum(1 for p in self._avoidant_compiled if p.search(text)) * 0.2
        )
        secure_score = float(
            sum(1 for p in self._secure_compiled if p.search(text)) * 0.2
        )

        # Prosodic cues
        if audio.f0_std > 40.0 and audio.speech_rate > 3.0:
            # High F0 variance + fast speech → anxious
            anxious_score += 0.3
        elif audio.f0_std < 10.0 and audio.rms < 0.01:
            # Flat monotone + low energy → avoidant
            avoidant_score += 0.3
        elif 10.0 <= audio.f0_std <= 40.0 and audio.pause_count < 5:
            # Moderate F0 variance + low pause count → secure
            secure_score += 0.2

        # Non-literal cue: anxious pattern
        if audio.vocal_cues.get("has_voice_rise") and audio.vocal_cues.get("has_hedging_pauses"):
            anxious_score += 0.15

        total = anxious_score + avoidant_score + secure_score
        if total < 0.05:
            return "secure", 0.3  # default with low confidence

        # Normalize
        anxious_score /= total
        avoidant_score /= total
        secure_score /= total

        if anxious_score >= avoidant_score and anxious_score >= secure_score:
            return "anxious", float(min(1.0, anxious_score))
        elif avoidant_score >= secure_score:
            return "avoidant", float(min(1.0, avoidant_score))
        else:
            return "secure", float(min(1.0, secure_score))

    def _detect_love_languages(self, text: str) -> Dict[str, float]:
        """Score each love language based on linguistic signals in text."""
        scores: Dict[str, float] = {}
        for lang, patterns in self._love_language_compiled.items():
            matches = sum(1 for p in patterns if p.search(text))
            scores[lang] = float(min(1.0, matches * 0.35))
        return scores

    def _detect_non_literal(
        self,
        text: str,
        audio: ProsodicFeatures,
    ) -> Tuple[bool, str, str, str]:
        """
        Detect incongruence between literal text meaning and prosodic signals.
        Returns: (detected, matched_phrase, actual_state, probable_meaning)
        """
        for phrase_pattern, prosodic_cue, actual_state, probable_meaning in INCONGRUENCE_PHRASES:
            if re.search(phrase_pattern, text, re.IGNORECASE):
                # Check if prosodic cue matches
                cue_matches = self._check_prosodic_cue(prosodic_cue, audio)
                if cue_matches:
                    matched = re.search(phrase_pattern, text, re.IGNORECASE)
                    return True, matched.group(0) if matched else phrase_pattern, actual_state, probable_meaning
        return False, "", "", ""

    def _check_prosodic_cue(self, cue: str, audio: ProsodicFeatures) -> bool:
        """Verify if a prosodic cue condition is met by the audio features."""
        if cue == "stressed_prosody":
            return audio.f0_std > 25.0 or audio.energy_std > 0.01
        elif cue == "high_energy":
            return audio.energy_mean > 0.02
        elif cue == "rising_f0":
            return audio.vocal_cues.get("has_voice_rise", False)
        elif cue == "flat_affect":
            return audio.f0_std < 8.0 and audio.energy_std < 0.005
        elif cue == "high_f0_variance":
            return audio.f0_std > 35.0
        elif cue == "minimal_energy":
            return audio.rms < 0.003
        elif cue == "fast_speech_rate":
            return audio.speech_rate > 4.0
        elif cue == "clipped_tone":
            return audio.pause_count < 2 and audio.duration_seconds < 3.0
        # Default: accept if at least some audio signal present
        return audio.rms > 0.001

    def _infer_communication_style(
        self,
        result: ClassificationResult,
        text: str,
        audio: ProsodicFeatures,
    ) -> str:
        """Infer overall communication style from combined signals."""
        if result.gottman_horsemen.get("stonewalling", 0.0) > 0.4:
            return "withdrawn"
        if result.non_literal_detected:
            return "indirect"
        if audio.vocal_cues.get("has_hedging_pauses"):
            return "hedging"
        return "direct"

    def _compute_distress_score(
        self,
        horsemen: Dict[str, float],
        attachment_pattern: str,
        audio: ProsodicFeatures,
        visual: Optional[VisualFeatures],
        emotion_scores: Dict[str, float],
    ) -> float:
        """
        Compute an aggregate distress score (0.0–1.0) from all modalities.
        Weights: Horsemen (0.4) + Attachment (0.2) + Audio (0.2) + Emotion (0.2)
        """
        # Horsemen component (max of four)
        horsemen_component = max(horsemen.values()) if horsemen else 0.0

        # Attachment component
        attachment_component = {
            "anxious": 0.6,
            "avoidant": 0.4,
            "secure": 0.1,
        }.get(attachment_pattern, 0.1)

        # Audio component: high F0 variance + high energy
        audio_stress = 0.0
        if audio.f0_std > 40.0:
            audio_stress += 0.4
        if audio.energy_mean > 0.05:
            audio_stress += 0.3
        if audio.vocal_cues.get("has_emotional_stress"):
            audio_stress += 0.3
        audio_component = min(1.0, audio_stress)

        # Emotion component: weight negative emotions
        emotion_component = 0.0
        if emotion_scores:
            negative_weight = (
                emotion_scores.get("anger", 0.0) * 1.0
                + emotion_scores.get("disgust", 0.0) * 0.9
                + emotion_scores.get("fear", 0.0) * 0.8
                + emotion_scores.get("sadness", 0.0) * 0.7
            )
            emotion_component = min(1.0, negative_weight)

        distress = (
            horsemen_component * 0.4
            + attachment_component * 0.2
            + audio_component * 0.2
            + emotion_component * 0.2
        )
        return float(min(1.0, distress))

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for efficiency."""
        flags = re.IGNORECASE
        self._criticism_compiled = [re.compile(p, flags) for p in CRITICISM_PATTERNS]
        self._contempt_compiled = [re.compile(p, flags) for p in CONTEMPT_PATTERNS]
        self._defensiveness_compiled = [re.compile(p, flags) for p in DEFENSIVENESS_PATTERNS]
        self._stonewalling_compiled = [re.compile(p, flags) for p in STONEWALLING_PATTERNS]
        self._anxious_compiled = [re.compile(p, flags) for p in ANXIOUS_PATTERNS]
        self._avoidant_compiled = [re.compile(p, flags) for p in AVOIDANT_PATTERNS]
        self._secure_compiled = [re.compile(p, flags) for p in SECURE_PATTERNS]
        self._love_language_compiled = {
            lang: [re.compile(p, flags) for p in patterns]
            for lang, patterns in LOVE_LANGUAGE_PATTERNS.items()
        }
