"""speech_processor.py — ASR + diarization + emotion for the input side.

Turns an ingested audio capture into the text + affect signals the dialogue
engine needs:
  * Whisper-large-v3 streaming ASR (via HFModelManager) -> transcript + timing.
  * pyannote diarization (optional) -> who-spoke-when for multi-speaker rooms.
  * emotion-distilroberta over the transcript -> dominant emotion (later mapped
    to a Live2D expression on the output side).

For text-only turns the ASR stage is skipped and emotion is computed directly.

Quality gate: a transcript must be non-empty after trimming; an empty Whisper
result (e.g. pure noise) is surfaced as ``low_confidence`` so the orchestrator
can ask the user to repeat instead of hallucinating a reply.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger("vtuber.speech_processor")


@dataclass
class SpeechResult:
    transcript: str
    language: str = "en"
    emotion: str = "neutral"
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    speakers: List[Dict[str, Any]] = field(default_factory=list)
    asr_latency_ms: float = 0.0
    asr_fallback: bool = False
    confident: bool = True
    reason: str = "ok"


class SpeechProcessor:
    def __init__(self, hf_manager=None, enable_diarization: bool = False,
                 min_transcript_chars: int = 1):
        self.hf = hf_manager
        self.enable_diarization = enable_diarization
        self.min_transcript_chars = min_transcript_chars

    def process_text(self, text: str) -> SpeechResult:
        emotion, scores = self._emotion(text)
        return SpeechResult(transcript=text.strip(), emotion=emotion,
                            emotion_scores=scores, confident=bool(text.strip()))

    def process_audio(self, audio_path: str, language: Optional[str] = None) -> SpeechResult:
        if self.hf is None:
            return SpeechResult(transcript="", confident=False,
                                reason="no HF manager for ASR", asr_fallback=True)
        start = time.time()
        asr = self.hf.transcribe(audio_path, language=language, word_timestamps=True)
        latency = (time.time() - start) * 1000.0
        transcript = asr.get("text", "").strip()
        speakers = []
        if self.enable_diarization:
            try:
                speakers = self.hf.diarize(audio_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("diarization skipped: %s", exc)

        result = SpeechResult(
            transcript=transcript, language=asr.get("language", "en"),
            speakers=speakers, asr_latency_ms=latency,
            asr_fallback=asr.get("fallback", False))

        if len(transcript) < self.min_transcript_chars:
            result.confident = False
            result.reason = "empty/low-confidence transcript"
            result.emotion, result.emotion_scores = "neutral", {"neutral": 1.0}
            return result

        result.emotion, result.emotion_scores = self._emotion(transcript)
        result.reason = "ok"
        return result

    def _emotion(self, text: str):
        if self.hf is None or not text.strip():
            return "neutral", {"neutral": 1.0}
        try:
            scores = self.hf.detect_emotion(text)
            dominant = max(scores, key=scores.get) if scores else "neutral"
            return dominant, scores
        except Exception as exc:  # noqa: BLE001
            logger.debug("emotion detection failed: %s", exc)
            return "neutral", {"neutral": 1.0}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sp = SpeechProcessor()
    print(sp.process_text("I'm so excited for today's stream!!"))
