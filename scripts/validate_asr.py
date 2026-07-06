"""Validate Whisper-large-v3 (or faster-whisper) ASR WER on a held-out clip set.

Phase 3 deliverable: confirm the streaming ASR model meets the <=10% WER target
on a held-out set before the pipeline trusts its transcripts.

Manifest format (JSONL / CSV / JSON): rows of
    {"audio": "clip.wav", "reference": "ground truth transcript",
     "language": "en", "emotion": "neutral"}

The script transcribes every clip with the configured ASR backend, computes
word error rate with benchmark_runner.word_error_rate, aggregates per-clip and
mean WER, records the run in the memory store, and writes a Markdown report
with a PASS/FAIL verdict against the quality gate.

Usage:
    python -m scripts.validate_asr --manifest clips.jsonl --label whisper-v3
    python -m scripts.validate_asr --manifest clips.csv --backend faster-whisper

No models are pulled by this script on its own; it will surface a clear
fallback / error when the model is not present, so it can run as a smoke test.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from typing import List, Dict, Any

from scripts._common import build_orchestrator, load_manifest, write_report, ROOT

logger = logging.getLogger("vtuber.validate_asr")

GATE_MAX_WER = 0.10  # CLAUDE.md folder-1 quality gate


def run(manifest_path: str, label: str, backend: str = None,
        compute: str = None, report_path: str = None) -> Dict[str, Any]:
    orch = build_orchestrator()
    hf = orch.hf()
    if backend:
        hf.set_asr_backend(backend, compute=compute)
    rows = load_manifest(manifest_path)

    per_clip: List[Dict[str, Any]] = []
    wers: List[float] = []
    fallbacks = 0
    for i, row in enumerate(rows):
        if not row["audio"]:
            per_clip.append({"index": i, "error": "no audio path"})
            continue
        result = hf.transcribe(row["audio"], language=row.get("language"),
                               word_timestamps=False)
        if result.get("fallback"):
            fallbacks += 1
        hyp = result.get("text", "").strip()
        wer = orch.bench().word_error_rate(row["reference"], hyp) if row["reference"] else None
        if wer is not None:
            wers.append(wer)
        per_clip.append({
            "index": i,
            "audio": row["audio"],
            "reference": row["reference"],
            "hypothesis": hyp,
            "wer": round(wer, 4) if wer is not None else None,
            "backend": result.get("backend", hf.asr_backend),
            "fallback": result.get("fallback", False),
            "language": result.get("language"),
            "error": result.get("error"),
        })

    mean_wer = round(statistics.mean(wers), 4) if wers else None
    p50 = round(statistics.median(wers), 4) if wers else None
    max_wer = round(max(wers), 4) if wers else None
    verdict = (mean_wer is not None and mean_wer <= GATE_MAX_WER)

    metrics = {
        "asr_wer": mean_wer,
        "asr_wer_p50": p50,
        "asr_wer_max": max_wer,
        "samples": len(rows),
        "transcribed": len(wers),
        "fallbacks": fallbacks,
        "backend": hf.asr_backend,
    }
    try:
        orch.memory().save_benchmark(label, {"asr_wer": mean_wer or 0.0},
                                      notes=f"validation run, backend={hf.asr_backend}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("benchmark persist failed: %s", exc)

    if report_path:
        lines = [
            f"- Manifest: `{manifest_path}`",
            f"- Backend: `{hf.asr_backend}` (compute={hf.faster_whisper_compute})",
            f"- Samples: {len(rows)} | transcribed: {len(wers)} | fallbacks: {fallbacks}",
            f"- Mean WER: {mean_wer} | p50: {p50} | max: {max_wer}",
            f"- Gate: mean WER <= {GATE_MAX_WER} -> "
            f"{'PASS' if verdict else 'FAIL' if mean_wer is not None else 'INCONCLUSIVE'}",
            "",
            "## Per-clip results",
            "| # | WER | Backend | Fallback | Reference | Hypothesis |",
            "|---|-----|---------|----------|-----------|------------|",
        ]
        for c in per_clip:
            if "error" in c and "wer" not in c:
                lines.append(f"| {c['index']} | - | - | - | - | _{c['error']}_ |")
                continue
            ref = (c["reference"] or "")[:60].replace("|", "/")
            hyp = (c["hypothesis"] or "")[:60].replace("|", "/")
            lines.append(f"| {c['index']} | {c['wer']} | {c['backend']} | "
                         f"{'yes' if c['fallback'] else 'no'} | {ref} | {hyp} |")
        write_report(report_path, f"ASR Validation - {label}", lines)
    return {"label": label, "verdict": verdict, "metrics": metrics, "per_clip": per_clip}


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Validate ASR WER on a held-out clip set.")
    ap.add_argument("--manifest", required=True, help="path to JSONL/CSV/JSON manifest")
    ap.add_argument("--label", default="whisper-v3-validation")
    ap.add_argument("--backend", default=None,
                    choices=("openai-whisper", "faster-whisper"),
                    help="override the ASR backend for this validation run")
    ap.add_argument("--compute", default=None,
                    help="faster-whisper compute type, e.g. int8_float16")
    ap.add_argument("--report", default=os.path.join(ROOT, "data", "reports",
                                                      "asr_validation.md"))
    args = ap.parse_args()
    out = run(args.manifest, args.label, args.backend, args.compute, args.report)
    print(f"mean WER: {out['metrics']['asr_wer']} -> "
          f"{'PASS' if out['verdict'] else 'FAIL/INCONCLUSIVE'} ({out['metrics']['backend']})")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
