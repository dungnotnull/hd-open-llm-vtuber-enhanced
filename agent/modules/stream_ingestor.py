"""stream_ingestor.py — real-time input capture for the VTuber turn loop.

Responsibilities:
  * Accept user input as a microphone capture, an audio file, or raw text.
  * Apply lightweight VAD (voice activity detection) gating so silence does not
    trigger an empty turn.
  * Validate audio quality (RMS / duration / sample-rate) before it reaches the
    ASR stage — a bad capture should fail fast, not waste a Whisper call.

Inputs : microphone device id, file path, or text string.
Outputs: ``IngestResult`` (audio path or text + quality metadata).
Tools called: sounddevice / soundfile / numpy (all optional, degrade to file or
text ingestion when unavailable).

Quality gate: RMS > silence_threshold AND 0.2s <= duration <= 30s.
"""

from __future__ import annotations

import os
import wave
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger("vtuber.stream_ingestor")


@dataclass
class IngestResult:
    modality: str                       # 'audio' | 'text'
    text: Optional[str] = None
    audio_path: Optional[str] = None
    duration_s: float = 0.0
    rms: float = 0.0
    sample_rate: int = 0
    valid: bool = True
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class StreamIngestor:
    def __init__(self, sample_rate: int = 16000, silence_threshold: float = 0.005,
                 min_duration_s: float = 0.2, max_duration_s: float = 30.0,
                 work_dir: str = "./data/captures"):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    # -- text -----------------------------------------------------------------
    def ingest_text(self, text: str) -> IngestResult:
        text = (text or "").strip()
        if not text:
            return IngestResult(modality="text", valid=False, reason="empty text")
        return IngestResult(modality="text", text=text, valid=True,
                            meta={"chars": len(text)})

    # -- file -----------------------------------------------------------------
    def ingest_file(self, path: str) -> IngestResult:
        if not os.path.exists(path):
            return IngestResult(modality="audio", valid=False,
                                reason=f"file not found: {path}")
        try:
            duration, rms, sr = self._probe_audio(path)
        except Exception as exc:  # noqa: BLE001
            return IngestResult(modality="audio", audio_path=path, valid=False,
                                reason=f"probe failed: {exc}")
        result = IngestResult(modality="audio", audio_path=path,
                              duration_s=duration, rms=rms, sample_rate=sr)
        self._apply_quality_gate(result)
        return result

    # -- microphone -----------------------------------------------------------
    def capture_microphone(self, seconds: float = 5.0,
                           device: Optional[int] = None) -> IngestResult:
        """Record from the default (or given) input device. Requires
        sounddevice + soundfile; otherwise returns an invalid result so the
        caller can fall back to text input."""
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
        except Exception as exc:  # noqa: BLE001
            return IngestResult(modality="audio", valid=False,
                                reason=f"audio capture deps unavailable: {exc}")
        try:
            logger.info("recording %.1fs from device %s ...", seconds, device)
            audio = sd.rec(int(seconds * self.sample_rate),
                           samplerate=self.sample_rate, channels=1,
                           dtype="float32", device=device)
            sd.wait()
            audio = audio.flatten()
            out_path = os.path.join(self.work_dir, f"capture_{int(seconds*1000)}_{id(audio)}.wav")
            sf.write(out_path, audio, self.sample_rate)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            result = IngestResult(modality="audio", audio_path=out_path,
                                  duration_s=seconds, rms=rms,
                                  sample_rate=self.sample_rate)
            self._apply_quality_gate(result)
            return result
        except Exception as exc:  # noqa: BLE001
            return IngestResult(modality="audio", valid=False,
                                reason=f"capture error: {exc}")

    # -- helpers --------------------------------------------------------------
    def _probe_audio(self, path: str):
        """Return (duration_s, rms, sample_rate). Uses soundfile when present,
        otherwise the stdlib wave module for PCM WAV."""
        try:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            duration = len(data) / float(sr)
            rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
            return duration, rms, sr
        except Exception:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                duration = n / float(sr)
                frames = wf.readframes(min(n, sr * 5))
            # crude RMS from raw 16-bit frames
            import struct
            count = len(frames) // 2
            if count == 0:
                return duration, 0.0, sr
            vals = struct.unpack("<" + "h" * count, frames[: count * 2])
            mean_sq = sum((v / 32768.0) ** 2 for v in vals) / count
            return duration, mean_sq ** 0.5, sr

    def _apply_quality_gate(self, result: IngestResult):
        if result.duration_s < self.min_duration_s:
            result.valid = False
            result.reason = f"too short ({result.duration_s:.2f}s)"
        elif result.duration_s > self.max_duration_s:
            result.valid = False
            result.reason = f"too long ({result.duration_s:.2f}s)"
        elif result.rms < self.silence_threshold:
            result.valid = False
            result.reason = f"silent (rms={result.rms:.5f})"
        else:
            result.valid = True
            result.reason = "ok"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ing = StreamIngestor()
    print(ing.ingest_text("Hello VTuber, how are you?"))
