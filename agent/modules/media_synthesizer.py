"""media_synthesizer.py — output side of the VTuber turn loop.

Takes the LLM's reply text plus the detected emotion and produces:
  * Speech audio via XTTS-v2 zero-shot voice cloning (HFModelManager).
  * A Live2D expression + motion command mapped from the emotion.
  * A viseme / lip-sync timeline aligned to the synthesized audio so the avatar
    mouth flaps in time with speech.

This is the module that makes the agent feel "alive": the same reply spoken
with a `joy` expression vs a `sadness` expression yields different avatar params.

Live2D mapping follows the common Cubism expression slots used by
Open-LLM-VTuber model configs (exp_01..exp_08 / motion groups).
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger("vtuber.media_synthesizer")

# emotion -> Live2D expression file + motion group + parameter overrides.
# Values follow Open-LLM-VTuber's emotion-tag convention so the produced
# command can be forwarded to the upstream frontend unchanged.
EMOTION_TO_LIVE2D = {
    "joy":      {"expression": "exp_joy",      "motion": "happy",   "params": {"ParamBrowLY": 0.6, "ParamMouthForm": 1.0}},
    "neutral":  {"expression": "exp_neutral",  "motion": "idle",    "params": {"ParamBrowLY": 0.0, "ParamMouthForm": 0.0}},
    "sadness":  {"expression": "exp_sad",      "motion": "sad",     "params": {"ParamBrowLY": -0.6, "ParamMouthForm": -0.8}},
    "anger":    {"expression": "exp_angry",    "motion": "angry",   "params": {"ParamBrowLY": -0.8, "ParamMouthForm": -0.4}},
    "fear":     {"expression": "exp_fear",     "motion": "surprise","params": {"ParamBrowLY": -0.3, "ParamEyeLOpen": 1.0}},
    "surprise": {"expression": "exp_surprise", "motion": "surprise","params": {"ParamEyeLOpen": 1.0, "ParamMouthOpenY": 0.8}},
    "disgust":  {"expression": "exp_disgust",  "motion": "angry",   "params": {"ParamBrowLY": -0.5, "ParamMouthForm": -0.6}},
}


@dataclass
class SynthesisResult:
    audio_path: Optional[str]
    expression: str
    motion: str
    live2d_params: Dict[str, float] = field(default_factory=dict)
    visemes: List[Dict[str, Any]] = field(default_factory=list)
    tts_latency_ms: float = 0.0
    tts_fallback: bool = False
    emotion: str = "neutral"


class MediaSynthesizer:
    def __init__(self, hf_manager=None, speaker_wav: Optional[str] = None,
                 out_dir: str = "./data/tts_out", language: str = "en"):
        self.hf = hf_manager
        self.speaker_wav = speaker_wav  # reference voice for cloning (6s clip)
        self.out_dir = out_dir
        self.language = language
        os.makedirs(out_dir, exist_ok=True)

    def synthesize(self, text: str, emotion: str = "neutral",
                   session_id: str = "session") -> SynthesisResult:
        emotion = emotion if emotion in EMOTION_TO_LIVE2D else "neutral"
        avatar = EMOTION_TO_LIVE2D[emotion]
        out_path = os.path.join(self.out_dir, f"{session_id}_{int(time.time()*1000)}.wav")

        tts_latency, fallback, visemes = 0.0, True, []
        if self.hf is not None and text.strip():
            start = time.time()
            meta = self.hf.synthesize(text, out_path, speaker_wav=self.speaker_wav,
                                      language=self.language)
            tts_latency = (time.time() - start) * 1000.0
            fallback = meta.get("fallback", True)
            visemes = meta.get("visemes", [])
        else:
            out_path = None
            visemes = self._fallback_visemes(text)

        return SynthesisResult(
            audio_path=out_path, expression=avatar["expression"],
            motion=avatar["motion"], live2d_params=dict(avatar["params"]),
            visemes=visemes, tts_latency_ms=tts_latency,
            tts_fallback=fallback, emotion=emotion)

    def build_frontend_command(self, result: SynthesisResult, reply_text: str) -> Dict[str, Any]:
        """Assemble the JSON payload Open-LLM-VTuber's frontend consumes to drive
        the avatar (audio url + expression + motion + lip-sync timeline)."""
        return {
            "type": "speak",
            "text": reply_text,
            "audio": result.audio_path,
            "expression": result.expression,
            "motion": result.motion,
            "live2d_params": result.live2d_params,
            "lip_sync": result.visemes,
            "emotion": result.emotion,
        }

    @staticmethod
    def _fallback_visemes(text: str) -> List[Dict[str, Any]]:
        words = (text or "").split()
        timeline, t = [], 0.0
        for w in words:
            dur = 0.06 * len(w) + 0.08
            timeline.append({"start": round(t, 3), "end": round(t + dur, 3),
                             "mouth_open": 0.5})
            t += dur
        return timeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ms = MediaSynthesizer()
    res = ms.synthesize("Yay, I'm so glad you're here today!", emotion="joy")
    print(res.expression, res.motion, res.live2d_params)
    print(ms.build_frontend_command(res, "Yay, I'm so glad you're here today!")["lip_sync"][:2])
