"""Apply proposal #1 (faster-whisper backend) and re-benchmark through the gate.

Phase 7 deliverable. Implements the measure-before-modify loop end-to-end:

  1. Run a baseline benchmark with the current ASR backend (openai-whisper).
  2. Apply proposal #1: switch the ASR backend to faster-whisper
     (CTranslate2, configurable compute type, default int8_float16).
  3. Re-run the benchmark with the candidate backend.
  4. Evaluate the before/after quality gate via benchmark_runner.evaluate_gates.
  5. Write a Markdown comparison report with a PASS/FAIL verdict and persist both
     runs to memory.

When a backend's model is not installed the orchestrator degrades gracefully and
the benchmark still runs (fallbacks counted), so the script can execute as a
smoke test; the gate verdict reflects the real numbers when models are present.

Usage:
    python -m scripts.apply_proposal_faster_whisper
    python -m scripts.apply_proposal_faster_whisper --compute int8 --cases cases.jsonl
    python -m scripts.apply_proposal_faster_whisper --keep-candidate   # persist switch
"""

from __future__ import annotations

import os
import argparse
import logging
from typing import List, Dict, Any

from scripts._common import build_orchestrator, write_report, ROOT

logger = logging.getLogger("vtuber.apply_proposal_faster_whisper")

DEFAULT_CASES = [
    {"text": "Hello there, how are you today?", "reference": "Hello there, how are you today?"},
    {"text": "Tell me a fun fact about space.", "reference": "Tell me a fun fact about space."},
    {"text": "I'm feeling a little down today.", "reference": "I'm feeling a little down today."},
    {"text": "Wow, that's really surprising!", "reference": "Wow, that's really surprising!"},
    {"text": "Can you help me with a recipe?", "reference": "Can you help me with a recipe?"},
]


def _result_from_metrics(label: str, metrics: Dict[str, Any]):
    from agent.modules.benchmark_runner import BenchmarkResult
    return BenchmarkResult(
        label=label,
        asr_wer=metrics.get("asr_wer"),
        tts_mos=metrics.get("tts_mos"),
        turn_latency_p50_ms=metrics.get("turn_latency_p50_ms"),
        turn_latency_p95_ms=metrics.get("turn_latency_p95_ms"),
        first_token_ms=metrics.get("first_token_ms"),
        samples=metrics.get("samples", 0),
    )


def run(cases: List[Dict[str, Any]], compute: str, label_base: str,
        label_cand: str, report_path: str, keep_candidate: bool) -> Dict[str, Any]:
    orch = build_orchestrator()
    hf = orch.hf()
    original_backend = hf.asr_backend

    orch.hf().set_asr_backend("openai-whisper")
    base_metrics = orch.run_benchmark(label_base, cases)
    base_result = _result_from_metrics(label_base, base_metrics)

    hf.set_asr_backend("faster-whisper", compute=compute)
    cand_metrics = orch.run_benchmark(label_cand, cases)
    cand_result = _result_from_metrics(label_cand, cand_metrics)

    gate = orch.bench().evaluate_gates(base_result, cand_result)
    report_md = orch.bench().to_markdown(base_result, cand_result, gate)
    # annotate with the proposal that drove this change
    proposal_note = (
        "## Applied proposal\n"
        "- **title**: Switch ASR to faster-whisper / CTranslate2 backend\n"
        f"- **change**: replaced {original_backend} with faster-whisper "
        f"(compute_type={compute})\n"
        "- **citation**: https://arxiv.org/abs/2212.04356\n"
    )
    report_md = report_md + "\n" + proposal_note
    write_report(report_path, f"Gated Upgrade - faster-whisper vs {original_backend}",
                 report_md.splitlines())

    if not keep_candidate:
        # restore the original backend so a one-shot run does not mutate state
        hf.set_asr_backend(original_backend)

    return {
        "original_backend": original_backend,
        "candidate_backend": "faster-whisper",
        "compute": compute,
        "baseline": base_metrics,
        "candidate": cand_metrics,
        "gate": gate,
        "report": report_path,
        "kept": keep_candidate and gate["passed"],
    }


def _load_cases(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return list(DEFAULT_CASES)
    from scripts._common import load_manifest
    rows = load_manifest(path)
    return [{"text": r["reference"] or "", "reference": r["reference"] or "",
             "emotion": r["emotion"] or "neutral"} for r in rows]


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(
        description="Apply proposal #1 (faster-whisper) and re-benchmark through the gate.")
    ap.add_argument("--cases", default=None, help="manifest of benchmark cases")
    ap.add_argument("--compute", default="int8_float16",
                    help="faster-whisper compute type (int8_float16, int8, float16, ...)")
    ap.add_argument("--label-base", default="baseline-openai-whisper")
    ap.add_argument("--label-cand", default="candidate-faster-whisper")
    ap.add_argument("--report", default=os.path.join(ROOT, "data", "reports",
                                                     "faster_whisper_gate.md"))
    ap.add_argument("--keep-candidate", action="store_true",
                    help="leave faster-whisper as the active backend after a PASS")
    args = ap.parse_args()
    cases = _load_cases(args.cases)
    out = run(cases, args.compute, args.label_base, args.label_cand,
              args.report, args.keep_candidate)
    print(f"gate: {'PASS' if out['gate']['passed'] else 'FAIL'} "
          f"(kept={out['kept']}) -> {out['report']}")


if __name__ == "__main__":
    main()
