"""
hf_model_manager.py — HuggingFace model manager for partner-decode-female-agent.

Singleton pattern. Lazy loading with 600-second idle unload timer.
CUDA auto-detect with CPU fallback.
Models: wav2vec2-base, emotion-distilroberta, whisper-large-v3,
        bge-large-en-v1.5, bge-reranker-large, bart-large-cnn
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Model registry
MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "emotion_classifier": "j-hartmann/emotion-english-distilroberta-base",
    "whisper": "openai/whisper-large-v3",
    "text_embedding": "BAAI/bge-large-en-v1.5",
    "reranker": "BAAI/bge-reranker-large",
    "summarizer": "facebook/bart-large-cnn",
}

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
IDLE_UNLOAD_SECONDS = 600.0


class HFModelManager:
    """
    Singleton HuggingFace model manager with lazy loading and idle unload.

    Usage:
        mgr = HFModelManager.get_instance()
        embedding = mgr.encode("some text")
        emotion = mgr.classify_emotion("I'm so upset")
        transcript = mgr.transcribe("audio.wav")
    """

    _instance: Optional["HFModelManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, cache_dir: str = "./models") -> "HFModelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(cache_dir=cache_dir)
        return cls._instance

    def __init__(self, cache_dir: str = "./models") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.cache_dir))

        # Device detection
        self.device = self._detect_device()
        logger.info("HFModelManager: device=%s, cache=%s", self.device, self.cache_dir)

        # Lazy model storage
        self._models: Dict[str, object] = {}
        self._processors: Dict[str, object] = {}
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_wav2vec2(
        self, audio: Union[np.ndarray, str], sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Extract 768-dimensional wav2vec2 embedding from audio.
        Resamples to 16kHz if needed. Returns mean-pooled last_hidden_state.
        """
        try:
            import torch
            processor, model = self._load_wav2vec2()

            if isinstance(audio, str):
                import librosa
                audio, sample_rate = librosa.load(audio, sr=16000, mono=True)
            elif sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean-pool over time dimension: [1, T, 768] → [768]
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            self._reset_timer("wav2vec2")
            return embedding.astype(np.float32)
        except Exception as exc:
            logger.warning("wav2vec2 extraction failed: %s", exc)
            return None

    def classify_emotion(self, text: str) -> Dict[str, float]:
        """
        Classify text into 7 emotion scores using emotion-distilroberta.
        Returns dict: {anger, disgust, fear, joy, neutral, sadness, surprise} → float
        """
        try:
            classifier = self._load_emotion_classifier()
            results = classifier(text, top_k=None, truncation=True, max_length=512)
            # results is a list of {label: str, score: float}
            scores = {item["label"]: round(item["score"], 4) for item in results}
            # Ensure all 7 labels present
            for label in EMOTION_LABELS:
                scores.setdefault(label, 0.0)
            self._reset_timer("emotion_classifier")
            return scores
        except Exception as exc:
            logger.warning("Emotion classification failed: %s", exc)
            return self._tfidf_emotion_fallback(text)

    def transcribe(
        self, audio_path: str, language: str = "en"
    ) -> Dict:
        """
        Transcribe audio file using Whisper-large-v3.
        Returns dict: {text: str, language: str, segments: [...]}
        """
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            pipe = self._load_whisper()
            result = pipe(
                audio_path,
                return_timestamps=True,
                generate_kwargs={"language": language, "task": "transcribe"},
            )
            self._reset_timer("whisper")
            return {
                "text": result.get("text", ""),
                "language": language,
                "chunks": result.get("chunks", []),
            }
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s — returning empty", exc)
            return {"text": "", "language": language, "chunks": []}

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string into a dense vector (bge-large-en-v1.5)."""
        model = self._load_text_embedding()
        try:
            embedding = model.encode(text, normalize_embeddings=normalize)
            self._reset_timer("text_embedding")
            return embedding.astype(np.float32)
        except Exception as exc:
            logger.warning("Text embedding failed: %s — using TF-IDF fallback", exc)
            return self._tfidf_fallback_encode(text)

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode multiple texts into a 2D array of embeddings."""
        model = self._load_text_embedding()
        try:
            embeddings = model.encode(texts, normalize_embeddings=normalize, batch_size=32)
            self._reset_timer("text_embedding")
            return embeddings.astype(np.float32)
        except Exception as exc:
            logger.warning("Batch embedding failed: %s — using TF-IDF fallback", exc)
            return np.vstack([self._tfidf_fallback_encode(t) for t in texts])

    def rerank(
        self, query: str, passages: List[str], top_k: int = 3
    ) -> List[Dict]:
        """
        Rerank passages using bge-reranker-large cross-encoder.
        Returns list of {passage, score, rank}.
        """
        try:
            from sentence_transformers import CrossEncoder
            model = self._load_reranker()
            pairs = [(query, p) for p in passages]
            scores = model.predict(pairs)
            ranked = sorted(
                zip(passages, scores), key=lambda x: x[1], reverse=True
            )[:top_k]
            self._reset_timer("reranker")
            return [
                {"passage": p, "score": float(s), "rank": i + 1}
                for i, (p, s) in enumerate(ranked)
            ]
        except Exception as exc:
            logger.warning("Reranker failed: %s — returning unranked", exc)
            return [{"passage": p, "score": 0.0, "rank": i + 1} for i, p in enumerate(passages[:top_k])]

    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize long text using BART-large-CNN."""
        try:
            summarizer = self._load_summarizer()
            result = summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
                truncation=True,
            )
            self._reset_timer("summarizer")
            return result[0]["summary_text"] if result else text[:max_length]
        except Exception as exc:
            logger.warning("Summarization failed: %s — truncating", exc)
            return text[:max_length] + "..."

    def preload(self, model_names: Optional[List[str]] = None) -> None:
        """Eagerly load specified models (useful for warm-up at startup)."""
        names = model_names or ["text_embedding", "emotion_classifier"]
        for name in names:
            try:
                getattr(self, f"_load_{name.replace('-', '_')}")()
                logger.info("Preloaded model: %s", name)
            except Exception as exc:
                logger.warning("Preload failed for %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Lazy Loaders
    # ------------------------------------------------------------------

    def _load_wav2vec2(self):
        """Lazy-load wav2vec2-base processor and model."""
        with self._lock:
            if "wav2vec2" not in self._models:
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                import torch
                model_id = os.environ.get("WAV2VEC2_MODEL", MODEL_IDS["wav2vec2"])
                logger.info("Loading %s…", model_id)
                processor = Wav2Vec2Processor.from_pretrained(
                    model_id, cache_dir=str(self.cache_dir)
                )
                model = Wav2Vec2Model.from_pretrained(
                    model_id, cache_dir=str(self.cache_dir)
                ).to(self.device)
                model.eval()
                self._processors["wav2vec2"] = processor
                self._models["wav2vec2"] = model
                self._start_idle_timer("wav2vec2")
            return self._processors["wav2vec2"], self._models["wav2vec2"]

    def _load_emotion_classifier(self):
        """Lazy-load emotion-distilroberta pipeline."""
        with self._lock:
            if "emotion_classifier" not in self._models:
                from transformers import pipeline
                model_id = os.environ.get("EMOTION_MODEL", MODEL_IDS["emotion_classifier"])
                logger.info("Loading %s…", model_id)
                pipe = pipeline(
                    "text-classification",
                    model=model_id,
                    return_all_scores=True,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)},
                )
                self._models["emotion_classifier"] = pipe
                self._start_idle_timer("emotion_classifier")
            return self._models["emotion_classifier"]

    def _load_whisper(self):
        """Lazy-load Whisper-large-v3 pipeline."""
        with self._lock:
            if "whisper" not in self._models:
                import torch
                from transformers import pipeline
                model_id = os.environ.get("WHISPER_MODEL", MODEL_IDS["whisper"])
                logger.info("Loading %s… (this may take a few minutes)", model_id)
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    torch_dtype=torch_dtype,
                    device=self.device,
                    model_kwargs={"cache_dir": str(self.cache_dir)},
                )
                self._models["whisper"] = pipe
                self._start_idle_timer("whisper")
            return self._models["whisper"]

    def _load_text_embedding(self):
        """Lazy-load bge-large-en-v1.5 SentenceTransformer."""
        with self._lock:
            if "text_embedding" not in self._models:
                try:
                    from sentence_transformers import SentenceTransformer
                    model_id = os.environ.get("EMBEDDING_MODEL", MODEL_IDS["text_embedding"])
                    logger.info("Loading %s…", model_id)
                    model = SentenceTransformer(model_id, cache_folder=str(self.cache_dir))
                    if self.device == "cuda":
                        model = model.to(self.device)
                    self._models["text_embedding"] = model
                    self._start_idle_timer("text_embedding")
                except ImportError:
                    logger.warning("sentence-transformers not installed — using TF-IDF fallback")
                    self._models["text_embedding"] = None
            return self._models["text_embedding"]

    def _load_reranker(self):
        """Lazy-load bge-reranker-large CrossEncoder."""
        with self._lock:
            if "reranker" not in self._models:
                from sentence_transformers import CrossEncoder
                model_id = os.environ.get("RERANKER_MODEL", MODEL_IDS["reranker"])
                logger.info("Loading %s…", model_id)
                model = CrossEncoder(model_id, max_length=512, device=self.device)
                self._models["reranker"] = model
                self._start_idle_timer("reranker")
            return self._models["reranker"]

    def _load_summarizer(self):
        """Lazy-load BART-large-CNN summarization pipeline."""
        with self._lock:
            if "summarizer" not in self._models:
                from transformers import pipeline
                model_id = os.environ.get("SUMMARIZER_MODEL", MODEL_IDS["summarizer"])
                logger.info("Loading %s…", model_id)
                pipe = pipeline(
                    "summarization",
                    model=model_id,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)},
                )
                self._models["summarizer"] = pipe
                self._start_idle_timer("summarizer")
            return self._models["summarizer"]

    # ------------------------------------------------------------------
    # Idle Unload
    # ------------------------------------------------------------------

    def _start_idle_timer(self, name: str) -> None:
        """Start a 600-second timer to unload the model if idle."""
        self._cancel_timer(name)
        timer = threading.Timer(IDLE_UNLOAD_SECONDS, self._unload_model, args=[name])
        timer.daemon = True
        timer.start()
        self._timers[name] = timer

    def _cancel_timer(self, name: str) -> None:
        """Cancel existing idle timer for a model."""
        timer = self._timers.pop(name, None)
        if timer:
            timer.cancel()

    def _reset_timer(self, name: str) -> None:
        """Reset idle timer after model use (keeps model warm during active sessions)."""
        if name in self._models:
            self._start_idle_timer(name)

    def _unload_model(self, name: str) -> None:
        """Unload a model from memory after idle timeout."""
        with self._lock:
            if name in self._models:
                try:
                    import torch
                    model = self._models.pop(name, None)
                    processor = self._processors.pop(name, None)
                    del model, processor
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    logger.info("Unloaded idle model: %s", name)
                except Exception as exc:
                    logger.debug("Error unloading %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Fallback Methods
    # ------------------------------------------------------------------

    def _tfidf_fallback_encode(self, text: str) -> np.ndarray:
        """
        Simple TF-IDF bag-of-words encoding as fallback when sentence-transformers unavailable.
        Returns a 300-dimensional sparse vector.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=300)
            vec = vectorizer.fit_transform([text]).toarray()[0]
            return vec.astype(np.float32)
        except ImportError:
            # Ultra-fallback: random unit vector
            vec = np.random.randn(300).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-8)

    def _tfidf_emotion_fallback(self, text: str) -> Dict[str, float]:
        """Simple keyword-based emotion fallback when model is unavailable."""
        text_lower = text.lower()
        scores = {label: 0.0 for label in EMOTION_LABELS}
        keyword_map = {
            "anger": ["angry", "furious", "mad", "rage", "hate"],
            "fear": ["scared", "afraid", "fear", "terrified", "anxious"],
            "joy": ["happy", "great", "wonderful", "love", "amazing"],
            "sadness": ["sad", "crying", "hurt", "devastated", "miss"],
            "disgust": ["disgusting", "gross", "awful", "terrible"],
            "surprise": ["wow", "omg", "unbelievable", "shocked"],
        }
        for emotion, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                scores[emotion] = 0.6
        # Normalize
        total = sum(scores.values())
        if total == 0:
            scores["neutral"] = 1.0
        else:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    # ------------------------------------------------------------------
    # Device Detection
    # ------------------------------------------------------------------

    def _detect_device(self) -> str:
        """Detect best available device: cuda > mps > cpu."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA detected: %s", torch.cuda.get_device_name(0))
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Apple MPS detected")
                return "mps"
        except ImportError:
            pass
        logger.info("No GPU detected — using CPU")
        return "cpu"
