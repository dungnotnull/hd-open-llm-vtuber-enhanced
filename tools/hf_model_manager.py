"""HuggingFace model manager for open-llm-vtuber-enhanced.

Centralizes lazy loading, caching, and graceful fallback for every pretrained
model the VTuber pipeline relies on. Singleton so that a single GPU resident
copy is shared across modules (ASR, TTS, emotion, retrieval).

Model registry (per CLAUDE.md "HuggingFace-first" principle):
  speech_recognition  : openai/whisper-large-v3            (streaming ASR)
  speaker_diarization : pyannote/speaker-diarization-3.1   (multi-speaker)
  tts_voice_cloning   : coqui/XTTS-v2                       (zero-shot TTS)
  emotion_detection   : j-hartmann/emotion-english-distilroberta-base
  text_embedding      : BAAI/bge-large-en-v1.5             (paper retrieval)
  reranking           : BAAI/bge-reranker-large            (proposal rerank)
  summarization       : facebook/bart-large-cnn            (abstract summary)

ASR backend selection:
  Set ``VTUBER_ASR_BACKEND`` (or pass ``asr_backend``) to one of
  ``"openai-whisper"`` (reference implementation) or ``"faster-whisper"``
  (CTranslate2 int8_float16, proposal #1). The manager exposes the same
  ``transcribe()`` contract regardless of backend, so the orchestrator can
  swap backends for benchmarking without touching call sites.

Every accessor degrades gracefully: if torch / transformers / the model is
unavailable, a deterministic heuristic fallback is returned so the agent keeps
running (CLAUDE.md design principle #11, graceful degradation).
"""

from __future__ import annotations

import os
import time
import math
import hashlib
import logging
import threading
from typing import Optional, List, Dict, Any

logger = logging.getLogger("vtuber.hf_models")

DEFAULT_MODELS = {
    "speech_recognition":  "openai/whisper-large-v3",
    "speaker_diarization": "pyannote/speaker-diarization-3.1",
    "tts_voice_cloning":   "coqui/XTTS-v2",
    "emotion_detection":   "j-hartmann/emotion-english-distilroberta-base",
    "text_embedding":      "BAAI/bge-large-en-v1.5",
    "reranking":           "BAAI/bge-reranker-large",
    "summarization":       "facebook/bart-large-cnn",
}

_IDLE_UNLOAD_SECONDS = 600  # unload a model after 10 min idle to free VRAM

VALID_ASR_BACKENDS = ("openai-whisper", "faster-whisper")
DEFAULT_ASR_BACKEND = "openai-whisper"
# CTranslate2 compute type used when faster-whisper is selected (proposal #1).
DEFAULT_FASTER_WHISPER_COMPUTE = "int8_float16"


class HFModelManager:
    """Singleton lazy-loading registry for HuggingFace models."""

    _instance: Optional["HFModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_dir: Optional[str] = None, device: Optional[str] = None,
                 asr_backend: Optional[str] = None,
                 faster_whisper_compute: Optional[str] = None):
        if getattr(self, "_initialized", False):
            # allow live backend/compute overrides on the singleton
            if asr_backend:
                self.asr_backend = asr_backend if asr_backend in VALID_ASR_BACKENDS else DEFAULT_ASR_BACKEND
            if faster_whisper_compute:
                self.faster_whisper_compute = faster_whisper_compute
            return
        self.cache_dir = cache_dir or os.getenv("HF_HOME", "./models")
        self.device = device or self._auto_device()
        env_backend = os.getenv("VTUBER_ASR_BACKEND", "").strip().lower()
        self.asr_backend = (asr_backend or env_backend or DEFAULT_ASR_BACKEND)
        if self.asr_backend not in VALID_ASR_BACKENDS:
            self.asr_backend = DEFAULT_ASR_BACKEND
        self.faster_whisper_compute = (faster_whisper_compute
                                       or os.getenv("VTUBER_FASTER_WHISPER_COMPUTE",
                                                    DEFAULT_FASTER_WHISPER_COMPUTE))
        self._models: Dict[str, Any] = {}
        self._last_used: Dict[str, float] = {}
        self._load_lock = threading.Lock()
        self._initialized = True
        logger.info("HFModelManager initialized (device=%s, cache=%s, asr=%s)",
                    self.device, self.cache_dir, self.asr_backend)

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    # -- generic load helpers -------------------------------------------------
    def _touch(self, key: str):
        self._last_used[key] = time.time()

    def unload_idle(self):
        now = time.time()
        for key in list(self._models):
            if now - self._last_used.get(key, now) > _IDLE_UNLOAD_SECONDS:
                logger.info("Unloading idle model: %s", key)
                self._models.pop(key, None)
                self._last_used.pop(key, None)
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def set_asr_backend(self, backend: str, compute: Optional[str] = None) -> str:
        """Swap the ASR backend at runtime (used when applying proposal #1 and
        re-benchmarking). The currently-loaded Whisper model is evicted so the
        next ``transcribe()`` reloads with the new backend."""
        if backend not in VALID_ASR_BACKENDS:
            raise ValueError(f"unknown ASR backend: {backend}")
        self.asr_backend = backend
        if compute:
            self.faster_whisper_compute = compute
        with self._load_lock:
            self._models.pop("whisper", None)
            self._last_used.pop("whisper", None)
        logger.info("ASR backend switched to %s (compute=%s)", backend, self.faster_whisper_compute)
        return self.asr_backend

    # ------------------------------------------------------------------ ASR --
    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   word_timestamps: bool = True) -> Dict[str, Any]:
        """Whisper-large-v3 transcription. Returns dict with text + segments.

        Backend-agnostic: openai-whisper or faster-whisper (CTranslate2) expose
        the same return contract. Falls back to an empty transcript marker when
        the model is unavailable so the pipeline can still proceed."""
        try:
            if self.asr_backend == "faster-whisper":
                result = self._transcribe_faster_whisper(
                    audio_path, language=language, word_timestamps=word_timestamps)
            else:
                result = self._transcribe_openai_whisper(
                    audio_path, language=language, word_timestamps=word_timestamps)
            self._touch("whisper")
            return {
                "text": result.get("text", "").strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", language or "en"),
                "backend": self.asr_backend,
                "fallback": False,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASR fallback (%s)", exc)
            return {"text": "", "segments": [], "language": language or "en",
                    "backend": self.asr_backend, "fallback": True, "error": str(exc)}

    def _transcribe_openai_whisper(self, audio_path: str, language: Optional[str],
                                   word_timestamps: bool) -> Dict[str, Any]:
        model = self._get_whisper_openai()
        result = model.transcribe(
            audio_path, language=language, word_timestamps=word_timestamps,
        )
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
                "words": [{"start": float(w.get("start", 0.0)),
                           "end": float(w.get("end", 0.0)),
                           "word": w.get("word", ""),
                           "probability": float(w.get("probability", 0.0))}
                          for w in (seg.get("words") or [])],
            })
        return {"text": result.get("text", ""), "segments": segments,
                "language": result.get("language", language or "en")}

    def _transcribe_faster_whisper(self, audio_path: str, language: Optional[str],
                                   word_timestamps: bool) -> Dict[str, Any]:
        model = self._get_whisper_faster()
        segments_iter, info = model.transcribe(
            audio_path, language=language, word_timestamps=word_timestamps,
            vad_filter=True, beam_size=5)
        segments = []
        full_text_parts = []
        for seg in segments_iter:
            words = []
            if word_timestamps:
                for w in (getattr(seg, "words", None) or []):
                    words.append({"start": float(w.start), "end": float(w.end),
                                  "word": w.word, "probability": float(w.probability)})
            seg_text = (seg.text or "").strip()
            full_text_parts.append(seg_text)
            segments.append({"start": float(seg.start), "end": float(seg.end),
                             "text": seg_text, "words": words})
        # faster-whisper streams lazily; consume to completion
        text = " ".join(p for p in full_text_parts if p).strip()
        return {"text": text, "segments": segments,
                "language": getattr(info, "language", language or "en")}

    def _get_whisper_openai(self):
        with self._load_lock:
            if "whisper_openai" not in self._models:
                import whisper  # openai-whisper
                logger.info("Loading Whisper large-v3 (openai-whisper) ...")
                self._models["whisper_openai"] = whisper.load_model(
                    "large-v3", device=self.device, download_root=self.cache_dir)
            self._models["whisper"] = self._models["whisper_openai"]
            return self._models["whisper_openai"]

    def _get_whisper_faster(self):
        with self._load_lock:
            if "whisper_faster" not in self._models:
                from faster_whisper import WhisperModel
                logger.info("Loading Whisper large-v3 (faster-whisper, %s) ...",
                            self.faster_whisper_compute)
                device = "cuda" if self.device == "cuda" else "cpu"
                self._models["whisper_faster"] = WhisperModel(
                    DEFAULT_MODELS["speech_recognition"], device=device,
                    compute_type=self.faster_whisper_compute,
                    download_root=self.cache_dir)
            self._models["whisper"] = self._models["whisper_faster"]
            return self._models["whisper_faster"]

    def _get_whisper(self):
        """Legacy accessor kept for backwards compatibility; routes to the
        currently selected backend."""
        if self.asr_backend == "faster-whisper":
            return self._get_whisper_faster()
        return self._get_whisper_openai()

    # ------------------------------------------------------------- diarize --
    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """pyannote speaker diarization -> list of {start, end, speaker}.

        Requires ``HF_TOKEN`` with access to the gated
        ``pyannote/speaker-diarization-3.1`` model. When the token or model is
        unavailable, a single-speaker fallback is returned so the turn loop
        keeps running."""
        if not os.getenv("HF_TOKEN"):
            logger.warning("HF_TOKEN not set; pyannote diarization requires a "
                           "gated-model access token. Single-speaker assumed.")
            return [{"start": 0.0, "end": 0.0, "speaker": "SPEAKER_00"}]
        try:
            pipeline = self._get_diarizer()
            diarization = pipeline(audio_path)
            out = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                out.append({"start": float(turn.start), "end": float(turn.end),
                            "speaker": str(speaker)})
            self._touch("diarizer")
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("diarization unavailable (%s) - single-speaker assumed", exc)
            return [{"start": 0.0, "end": 0.0, "speaker": "SPEAKER_00"}]

    def _get_diarizer(self):
        with self._load_lock:
            if "diarizer" not in self._models:
                from pyannote.audio import Pipeline
                logger.info("Loading pyannote diarization-3.1 ...")
                self._models["diarizer"] = Pipeline.from_pretrained(
                    DEFAULT_MODELS["speaker_diarization"],
                    use_auth_token=os.getenv("HF_TOKEN"))
                if self.device == "cuda":
                    try:
                        self._models["diarizer"].to(torch_device("cuda"))
                    except Exception:  # noqa: BLE001
                        pass
            return self._models["diarizer"]

    # ----------------------------------------------------------------- TTS --
    def synthesize(self, text: str, out_path: str,
                   speaker_wav: Optional[str] = None,
                   language: str = "en") -> Dict[str, Any]:
        """XTTS-v2 zero-shot TTS. Returns metadata incl. estimated visemes.

        Falls back to a silent/placeholder WAV when the model is unavailable so
        downstream lip-sync code still has a file to time against."""
        try:
            tts = self._get_xtts()
            tts.tts_to_file(text=text, file_path=out_path,
                            speaker_wav=speaker_wav, language=language)
            self._touch("xtts")
            return {"path": out_path, "fallback": False,
                    "visemes": self._estimate_visemes(text)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS fallback (%s)", exc)
            self._write_silence(out_path, seconds=max(1.0, len(text) / 15.0))
            return {"path": out_path, "fallback": True, "error": str(exc),
                    "visemes": self._estimate_visemes(text)}

    def _get_xtts(self):
        with self._load_lock:
            if "xtts" not in self._models:
                from TTS.api import TTS  # coqui-tts
                logger.info("Loading XTTS-v2 ...")
                self._models["xtts"] = TTS(
                    DEFAULT_MODELS["tts_voice_cloning"]).to(self.device)
            return self._models["xtts"]

    @staticmethod
    def _estimate_visemes(text: str) -> List[Dict[str, Any]]:
        """Cheap viseme timeline from text length (mouth open ratio per word).

        Real phoneme timing comes from the TTS engine when available; this
        deterministic estimate keeps the Live2D mouth flap alive in fallback."""
        words = text.split()
        timeline = []
        t = 0.0
        for w in words:
            dur = 0.06 * len(w) + 0.08
            open_ratio = min(1.0, 0.3 + 0.1 * sum(c in "aeiouAEIOU" for c in w))
            timeline.append({"start": round(t, 3), "end": round(t + dur, 3),
                             "mouth_open": round(open_ratio, 3)})
            t += dur
        return timeline

    @staticmethod
    def _write_silence(path: str, seconds: float, sample_rate: int = 22050):
        import wave
        import struct
        n = int(seconds * sample_rate)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(struct.pack("<h", 0) for _ in range(n)))

    # ------------------------------------------------------------- emotion --
    def detect_emotion(self, text: str) -> Dict[str, float]:
        """7-class emotion scores from distilroberta. Drives Live2D expression.

        Heuristic keyword fallback keeps expressions working offline."""
        if not text.strip():
            return {"neutral": 1.0}
        try:
            clf = self._get_emotion()
            raw = clf(text, top_k=None)
            self._touch("emotion")
            scores = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
            return {s["label"]: float(s["score"]) for s in scores}
        except Exception as exc:  # noqa: BLE001
            logger.debug("emotion fallback (%s)", exc)
            return self._heuristic_emotion(text)

    def _get_emotion(self):
        with self._load_lock:
            if "emotion" not in self._models:
                from transformers import pipeline
                logger.info("Loading emotion-distilroberta ...")
                dev = 0 if self.device == "cuda" else -1
                self._models["emotion"] = pipeline(
                    "text-classification", model=DEFAULT_MODELS["emotion_detection"],
                    device=dev, return_all_scores=True)
            return self._models["emotion"]

    @staticmethod
    def _heuristic_emotion(text: str) -> Dict[str, float]:
        t = text.lower()
        cues = {
            "joy": ["happy", "yay", "great", "love", "fun", "awesome", "haha", "!"],
            "anger": ["angry", "mad", "hate", "annoy", "ugh"],
            "sadness": ["sad", "cry", "sorry", "miss", "lonely"],
            "fear": ["scared", "afraid", "worry", "nervous"],
            "surprise": ["wow", "really", "whoa", "omg", "?!"],
            "disgust": ["gross", "eww", "disgust"],
        }
        scores = {k: 0.0 for k in cues}
        for emo, words in cues.items():
            scores[emo] = sum(t.count(w) for w in words)
        total = sum(scores.values())
        if total == 0:
            return {"neutral": 1.0}
        out = {k: v / total for k, v in scores.items() if v > 0}
        out["neutral"] = 0.1
        norm = sum(out.values())
        return {k: v / norm for k, v in out.items()}

    # ---------------------------------------------------------- embeddings --
    def encode(self, texts: List[str]) -> List[List[float]]:
        """BGE-large embeddings for paper retrieval. TF-IDF-ish fallback."""
        try:
            model = self._get_embedder()
            vecs = model.encode(texts, normalize_embeddings=True)
            self._touch("embedder")
            return [v.tolist() for v in vecs]
        except Exception as exc:  # noqa: BLE001
            logger.debug("embedding fallback (%s)", exc)
            return [self._hash_embed(t) for t in texts]

    def _get_embedder(self):
        with self._load_lock:
            if "embedder" not in self._models:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading BGE-large embedder ...")
                self._models["embedder"] = SentenceTransformer(
                    DEFAULT_MODELS["text_embedding"], device=self.device,
                    cache_folder=self.cache_dir)
            return self._models["embedder"]

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> List[float]:
        """Deterministic hashing embedding so retrieval still ranks sensibly."""
        vec = [0.0] * dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # ----------------------------------------------------------- reranking --
    def rerank(self, query: str, docs: List[str]) -> List[float]:
        try:
            ce = self._get_reranker()
            scores = ce.predict([(query, d) for d in docs])
            self._touch("reranker")
            return [float(s) for s in scores]
        except Exception as exc:  # noqa: BLE001
            logger.debug("rerank fallback (%s)", exc)
            q = set(query.lower().split())
            return [len(q & set(d.lower().split())) / (len(q) + 1) for d in docs]

    def _get_reranker(self):
        with self._load_lock:
            if "reranker" not in self._models:
                from sentence_transformers import CrossEncoder
                logger.info("Loading BGE-reranker ...")
                self._models["reranker"] = CrossEncoder(
                    DEFAULT_MODELS["reranking"], device=self.device)
            return self._models["reranker"]

    # -------------------------------------------------------- summarization --
    def summarize(self, text: str, max_length: int = 130) -> str:
        try:
            summarizer = self._get_summarizer()
            out = summarizer(text[:3500], max_length=max_length,
                             min_length=20, do_sample=False)
            self._touch("summarizer")
            return out[0]["summary_text"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("summary fallback (%s)", exc)
            sentences = text.replace("\n", " ").split(". ")
            return ". ".join(sentences[:2])[:max_length * 4]

    def _get_summarizer(self):
        with self._load_lock:
            if "summarizer" not in self._models:
                from transformers import pipeline
                logger.info("Loading BART-CNN summarizer ...")
                dev = 0 if self.device == "cuda" else -1
                self._models["summarizer"] = pipeline(
                    "summarization", model=DEFAULT_MODELS["summarization"], device=dev)
            return self._models["summarizer"]


def torch_device(name: str):
    try:
        import torch
        return torch.device(name)
    except Exception:  # noqa: BLE001
        return name


def get_manager() -> HFModelManager:
    return HFModelManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mgr = get_manager()
    print("device:", mgr.device, "asr_backend:", mgr.asr_backend)
    print("emotion:", mgr.detect_emotion("I'm so happy to see you today!!"))
    print("visemes:", mgr._estimate_visemes("hello world")[:2])
