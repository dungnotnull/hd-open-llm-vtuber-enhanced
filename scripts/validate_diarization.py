"""Validate the pyannote-3.1 speaker-diarization path (gated model, HF token).

Phase 3 deliverable: enable and exercise the multi-speaker diarization path so
the pipeline can distinguish speakers in a room, while degrading cleanly when
the gated model or the HuggingFace access token is not available.

Requirements to fully run:
  * ``HF_TOKEN`` env var with access to ``pyannote/speaker-diarization-3.1``
  * ``pyannote.audio`` installed (optional dependency, see requirements.txt)

Inputs:
  --audio        : path to a multi-speaker audio clip to diarize
  --expected     : optional JSONL manifest of ground-truth speaker turns for DER
                    {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"}

The script runs diarization via hf_model_manager.diarize, computes a simple
diarization error rate (DER) against the expected turns when provided, records
the run in memory, and writes a Markdown report.

Without an HF token the script reports the single-speaker fallback and exits
cleanly (it still runs as a smoke test of the gated-model path).

Usage:
    python -m scripts.validate_diarization --audio room.wav --expected turns.jsonl
    python -m scripts.validate_diarization --audio room.wav --label multi-speaker
"""

from __future__ import annotations

import os
import argparse
import logging
from typing import List, Dict, Any, Optional

from scripts._common import build_orchestrator, write_report, ROOT

logger = logging.getLogger("vtuber.validate_diarization")


def _load_turns(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    import json
    turns: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            turns.append(json.loads(line))
    return turns


def diarization_error_rate(reference: List[Dict[str, Any]],
                            hypothesis: List[Dict[str, Any]]) -> float:
    """Simple, alignment-free DER estimate based on the Jaccard mismatch.

    Maps each turn's [start,end) interval to its speaker and computes the
    fraction of total reference-speech time not covered by the same speaker.
    This is an estimate suitable for a CI-grade gate; pyannote's full DER
    scoring can be substituted where available. Returns 0.0 when no reference
    is provided."""
    if not reference:
        return 0.0
    ref_total = sum(t["end"] - t["start"] for t in reference if t["end"] > t["start"])
    if ref_total <= 0:
        return 0.0
    # union of intervals per speaker
    hyp_by_spk: Dict[str, List[tuple]] = {}
    for t in hypothesis:
        hyp_by_spk.setdefault(t.get("speaker", "SPEAKER_00"), []).append(
            (t["start"], t["end"]))
    correct = 0.0
    for t in reference:
        seg = (t["start"], t["end"])
        spk = t.get("speaker")
        # find overlapping hypothesis time with the same speaker label
        for hs, he in hyp_by_spk.get(spk, []):
            ov = max(0.0, min(seg[1], he) - max(seg[0], hs))
            correct += ov
            if ov >= (seg[1] - seg[0]):
                break
    return round(max(0.0, 1.0 - correct / ref_total), 4)


def run(audio: str, expected: Optional[str], label: str,
        report_path: Optional[str] = None) -> Dict[str, Any]:
    orch = build_orchestrator()
    hf = orch.hf()
    token_set = bool(os.getenv("HF_TOKEN"))
    turns = hf.diarize(audio)
    ref_turns = _load_turns(expected)
    der = diarization_error_rate(ref_turns, turns) if ref_turns else None
    speakers = sorted({t["speaker"] for t in turns})
    speech_time = sum(t["end"] - t["start"] for t in turns if t["end"] > t["start"])
    fallback = (len(turns) == 1 and turns[0]["end"] == 0.0)

    metrics = {
        "audio": audio,
        "hf_token_set": token_set,
        "speakers": len(speakers),
        "speaker_labels": speakers,
        "turns": len(turns),
        "speech_time_s": round(speech_time, 3),
        "der": der,
        "fallback": fallback,
    }
    try:
        orch.memory().save_benchmark(label, {"tts_mos": 0.0},
                                      notes=f"diarization validation, "
                                            f"speakers={len(speakers)}, "
                                            f"der={der}, fallback={fallback}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark persist failed: %s", exc)

    if report_path:
        lines = [
            f"- Audio: `{audio}`",
            f"- HF_TOKEN set: {'yes' if token_set else 'no (single-speaker fallback)'}",
            f"- Detected speakers: {len(speakers)} ({', '.join(speakers) or '-'})",
            f"- Speech turns: {len(turns)} | speech time: {metrics['speech_time_s']}s",
            f"- DER vs expected: {der if der is not None else 'n/a'}",
            f"- Fallback: {'yes' if fallback else 'no'}",
            "",
            "## Speaker turns",
            "| start | end | speaker |",
            "|-------|-----|---------|",
        ]
        for t in turns:
            lines.append(f"| {round(t['start'], 3)} | {round(t['end'], 3)} | "
                         f"{t['speaker']} |")
        write_report(report_path, f"Diarization Validation - {label}", lines)
    return {"label": label, "metrics": metrics, "turns": turns}


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Validate the pyannote diarization path.")
    ap.add_argument("--audio", required=True, help="path to audio clip to diarize")
    ap.add_argument("--expected", default=None,
                    help="optional JSONL ground-truth turns for DER scoring")
    ap.add_argument("--label", default="pyannote-diarization-validation")
    ap.add_argument("--report", default=os.path.join(ROOT, "data", "reports",
                                                      "diarization_validation.md"))
    args = ap.parse_args()
    out = run(args.audio, args.expected, args.label, args.report)
    m = out["metrics"]
    print(f"speakers: {m['speakers']} | turns: {m['turns']} | "
          f"DER: {m['der']} | fallback: {m['fallback']}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
