"""Validate XTTS-v2 zero-shot voice cloning from a 6-second reference voice.

Phase 3 deliverable: confirm the TTS engine can clone a voice from a 6s
reference clip and produce audible, well-formed audio across a held-out prompt
set, meeting the naturalness gate.

Inputs:
  --reference    : path to a ~6s clean WAV of the target voice (the cloning ref)
  --prompts       : JSONL/CSV/JSON manifest of prompts ({"text": "...", "emotion": "joy"})
                    if omitted, a built-in diverse prompt set is used
  --language      : ISO language code (default en)
  --out-dir       : where synthesized clips are written
  --label         : benchmark label for the memory store

The script synthesizes every prompt with XTTS-v2, measures audio duration /
energy / sample-rate of each output, computes the objective MOS proxy from
benchmark_runner, records the run in memory, and writes a Markdown report.

When XTTS-v2 is not installed the synthesis falls back to silence and the report
makes that explicit (the script still runs end-to-end as a smoke test).

Usage:
    python -m scripts.validate_tts --reference voice.wav --label xtts-clone-ref1
    python -m scripts.validate_tts --reference voice.wav --prompts prompts.jsonl
"""

from __future__ import annotations

import os
import argparse
import logging
import statistics
from typing import List, Dict, Any, Optional

from scripts._common import build_orchestrator, write_report, ROOT

logger = logging.getLogger("vtuber.validate_tts")

GATE_MIN_MOS = 4.0
DEFAULT_PROMPTS = [
    {"text": "Hello! It's so wonderful to meet you today.", "emotion": "joy"},
    {"text": "I'm really sorry to hear that, I'm here for you.", "emotion": "sadness"},
    {"text": "Whoa, that's amazing! Tell me more about it!", "emotion": "surprise"},
    {"text": "Let me think about that for a moment.", "emotion": "neutral"},
    {"text": "That makes me a little upset, but let's work through it.", "emotion": "anger"},
]


def _probe_wav(path: str) -> Dict[str, Any]:
    """Read duration / rms / sample-rate from a PCM WAV using stdlib."""
    import wave
    import struct
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            channels = wf.getnchannels()
            duration = n / float(sr)
            raw = wf.readframes(min(n, sr * 5))
        count = len(raw) // (wf.getsampwidth() or 2) if False else len(raw) // 2
        vals = struct.unpack("<" + "h" * count, raw[: count * 2]) if count else []
        rms = (sum((v / 32768.0) ** 2 for v in vals) / count) ** 0.5 if count else 0.0
        return {"duration_s": round(duration, 3), "rms": round(float(rms), 5),
                "sample_rate": sr, "channels": channels, "present": True}
    except Exception as exc:  # noqa: BLE001
        return {"duration_s": 0.0, "rms": 0.0, "sample_rate": 0,
                "channels": 0, "present": False, "error": str(exc)}


def run(reference: Optional[str], prompts: List[Dict[str, str]],
         language: str, out_dir: str, label: str,
         report_path: Optional[str] = None) -> Dict[str, Any]:
    orch = build_orchestrator()
    hf = orch.hf()
    os.makedirs(out_dir, exist_ok=True)

    per_prompt: List[Dict[str, Any]] = []
    fallbacks = 0
    durations: List[float] = []
    energies: List[float] = []
    for i, p in enumerate(prompts):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        emotion = p.get("emotion") or "neutral"
        out_path = os.path.join(out_dir, f"{label}_{i:03d}.wav")
        meta = hf.synthesize(text, out_path, speaker_wav=reference, language=language)
        probe = _probe_wav(out_path) if meta.get("path") else {"present": False}
        if meta.get("fallback"):
            fallbacks += 1
        if probe.get("present"):
            durations.append(probe["duration_s"])
            energies.append(probe["rms"])
        per_prompt.append({
            "index": i,
            "text": text,
            "emotion": emotion,
            "audio": out_path if probe.get("present") else None,
            "fallback": bool(meta.get("fallback")),
            "duration_s": probe.get("duration_s"),
            "rms": probe.get("rms"),
            "sample_rate": probe.get("sample_rate"),
            "error": meta.get("error"),
        })

    total = len(per_prompt) or 1
    success_ratio = (total - fallbacks) / total
    avg_audio_present = sum(1.0 for c in per_prompt if c["audio"]) / total
    mos_proxy = orch.bench().mos_proxy(fallbacks, total, avg_audio_present)
    # audible-energy bonus: clones that actually contain non-silent audio rate higher
    energy_score = (sum(1 for e in energies if e > 0.001) / len(energies)) if energies else 0.0
    mos_proxy = round(mos_proxy * (0.6 + 0.4 * energy_score), 3)
    verdict = mos_proxy >= GATE_MIN_MOS and fallbacks == 0

    metrics = {
        "tts_mos": mos_proxy,
        "samples": total,
        "fallbacks": fallbacks,
        "avg_duration_s": round(statistics.mean(durations), 3) if durations else None,
        "audible_ratio": energy_score,
        "reference": reference,
        "language": language,
    }
    try:
        orch.memory().save_benchmark(label, {"tts_mos": mos_proxy},
                                      notes=f"xtts clone validation, ref={reference}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark persist failed: %s", exc)

    if report_path:
        lines = [
            f"- Reference voice: `{reference or '(none)'}`",
            f"- Language: `{language}`",
            f"- Prompts: {total} | fallbacks: {fallbacks}",
            f"- Avg duration: {metrics['avg_duration_s']}s | audible ratio: {energy_score:.2f}",
            f"- MOS proxy: {mos_proxy}",
            f"- Gate: MOS >= {GATE_MIN_MOS} and 0 fallbacks -> "
            f"{'PASS' if verdict else 'FAIL'}",
            "",
            "## Per-prompt results",
            "| # | Emotion | Fallback | Duration (s) | RMS | Audio |",
            "|---|---------|----------|--------------|-----|-------|",
        ]
        for c in per_prompt:
            lines.append(f"| {c['index']} | {c['emotion']} | "
                         f"{'yes' if c['fallback'] else 'no'} | "
                         f"{c['duration_s']} | {c['rms']} | "
                         f"`{c['audio'] or '-'}` |")
        write_report(report_path, f"TTS Cloning Validation - {label}", lines)
    return {"label": label, "verdict": verdict, "metrics": metrics, "per_prompt": per_prompt}


def _load_prompts(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return list(DEFAULT_PROMPTS)
    from scripts._common import load_manifest  # reuse manifest loader
    rows = load_manifest(path)
    return [{"text": r["reference"], "emotion": r["emotion"] or "neutral"}
            for r in rows]


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Validate XTTS-v2 voice cloning from a 6s reference.")
    ap.add_argument("--reference", default=os.getenv("REFERENCE_VOICE_WAV"),
                    help="path to ~6s reference voice WAV")
    ap.add_argument("--prompts", default=None,
                    help="manifest of prompts (optional; built-in set used otherwise)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "tts_validation"))
    ap.add_argument("--label", default="xtts-v2-clone-validation")
    ap.add_argument("--report", default=os.path.join(ROOT, "data", "reports",
                                                     "tts_validation.md"))
    args = ap.parse_args()
    prompts = _load_prompts(args.prompts)
    out = run(args.reference, prompts, args.language, args.out_dir, args.label, args.report)
    print(f"MOS proxy: {out['metrics']['tts_mos']} -> "
          f"{'PASS' if out['verdict'] else 'FAIL'} ({out['metrics']['samples']} clips)")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
