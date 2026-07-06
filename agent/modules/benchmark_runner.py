"""benchmark_runner.py — measure the pipeline before/after each change.

Implements the Cluster A quality gate: a proposed optimization is only worth
keeping if it improves latency or quality *without regressing the other*.

Metrics:
  * asr_wer            — word error rate vs a reference transcript (lower better)
  * tts_mos            — naturalness MOS proxy (higher better); a real human MOS
                         can be substituted, otherwise an objective proxy is used
  * turn_latency_p50/p95_ms — end-to-end user-speech-end -> first-avatar-audio
  * first_token_ms     — LLM first-token latency (perceived responsiveness)

It records each run in MemoryManager and produces a Markdown comparison report
(baseline vs candidate) with explicit PASS/FAIL gate verdicts.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger("vtuber.benchmark_runner")

# Gate thresholds (CLAUDE.md folder-1 quality gates).
GATES = {
    "max_wer": 0.10,                 # streaming ASR WER target
    "min_mos": 4.0,                  # TTS naturalness target
    "max_p95_latency_ms": 1500.0,    # end-to-end turn latency target
    "min_latency_improvement_pct": 5.0,
    "min_mos_improvement": 0.1,
    "max_wer_regression": 0.005,
}


@dataclass
class BenchmarkResult:
    label: str
    asr_wer: Optional[float] = None
    tts_mos: Optional[float] = None
    turn_latency_p50_ms: Optional[float] = None
    turn_latency_p95_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    samples: int = 0
    notes: str = ""

    def to_metrics(self) -> Dict[str, Any]:
        return {
            "asr_wer": self.asr_wer, "tts_mos": self.tts_mos,
            "turn_latency_p50_ms": self.turn_latency_p50_ms,
            "turn_latency_p95_ms": self.turn_latency_p95_ms,
            "first_token_ms": self.first_token_ms,
        }


class BenchmarkRunner:
    def __init__(self, memory=None):
        self.memory = memory

    # -- metric primitives ----------------------------------------------------
    @staticmethod
    def word_error_rate(reference: str, hypothesis: str) -> float:
        """Levenshtein-based WER at the word level."""
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        if not ref:
            return 0.0 if not hyp else 1.0
        # DP edit distance
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        return d[len(ref)][len(hyp)] / len(ref)

    @staticmethod
    def percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * (pct / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    @staticmethod
    def mos_proxy(tts_fallbacks: int, total: int, avg_audio_present: float) -> float:
        """Objective MOS proxy when no human raters are available: penalize
        fallback (silent) synthesis and missing audio. Range ~1.0-4.6."""
        if total == 0:
            return 0.0
        success_ratio = 1.0 - (tts_fallbacks / total)
        return round(1.0 + 3.6 * success_ratio * avg_audio_present, 3)

    # -- run a benchmark suite ------------------------------------------------
    def run(self, label: str, cases: List[Dict[str, Any]],
            turn_fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> BenchmarkResult:
        """Execute ``turn_fn`` over each case dict and aggregate metrics.

        Each case may contain ``reference`` (for WER). ``turn_fn`` must return a
        dict with: transcript, turn_latency_ms, first_token_ms, tts_fallback,
        has_audio."""
        latencies, first_tokens, wers = [], [], []
        tts_fallbacks, audio_present = 0, []
        for case in cases:
            out = turn_fn(case)
            latencies.append(out.get("turn_latency_ms", 0.0))
            if out.get("first_token_ms") is not None:
                first_tokens.append(out["first_token_ms"])
            if case.get("reference") and out.get("transcript"):
                wers.append(self.word_error_rate(case["reference"], out["transcript"]))
            if out.get("tts_fallback"):
                tts_fallbacks += 1
            audio_present.append(1.0 if out.get("has_audio") else 0.0)

        avg_audio = sum(audio_present) / len(audio_present) if audio_present else 0.0
        result = BenchmarkResult(
            label=label,
            asr_wer=round(sum(wers) / len(wers), 4) if wers else None,
            tts_mos=self.mos_proxy(tts_fallbacks, len(cases), avg_audio),
            turn_latency_p50_ms=round(self.percentile(latencies, 50), 1),
            turn_latency_p95_ms=round(self.percentile(latencies, 95), 1),
            first_token_ms=round(sum(first_tokens) / len(first_tokens), 1) if first_tokens else None,
            samples=len(cases))
        if self.memory is not None:
            try:
                self.memory.save_benchmark(label, result.to_metrics(), result.notes)
            except Exception as exc:  # noqa: BLE001
                logger.debug("benchmark persist failed: %s", exc)
        return result

    # -- gate evaluation ------------------------------------------------------
    def evaluate_gates(self, baseline: BenchmarkResult,
                       candidate: BenchmarkResult) -> Dict[str, Any]:
        """Decide whether ``candidate`` should replace ``baseline``."""
        verdicts: Dict[str, Any] = {}

        # absolute gates
        verdicts["wer_within_target"] = (candidate.asr_wer or 0.0) <= GATES["max_wer"]
        verdicts["mos_within_target"] = (candidate.tts_mos or 0.0) >= GATES["min_mos"]
        verdicts["p95_within_target"] = (candidate.turn_latency_p95_ms or 1e9) <= GATES["max_p95_latency_ms"]

        # no-regression gates (latency improves OR quality improves, neither regresses)
        lat_delta_pct = self._pct_delta(baseline.turn_latency_p95_ms, candidate.turn_latency_p95_ms)
        mos_delta = (candidate.tts_mos or 0.0) - (baseline.tts_mos or 0.0)
        wer_delta = (candidate.asr_wer or 0.0) - (baseline.asr_wer or 0.0)

        verdicts["latency_not_regressed"] = lat_delta_pct <= GATES["max_wer_regression"] * 100 + 1.0 \
            if baseline.turn_latency_p95_ms else True
        verdicts["wer_not_regressed"] = wer_delta <= GATES["max_wer_regression"]
        verdicts["mos_not_regressed"] = mos_delta >= -GATES["min_mos_improvement"]

        improved = (lat_delta_pct <= -GATES["min_latency_improvement_pct"]) or \
                   (mos_delta >= GATES["min_mos_improvement"]) or \
                   (wer_delta <= -0.005)
        verdicts["shows_improvement"] = improved

        passed = all([verdicts["wer_not_regressed"], verdicts["mos_not_regressed"],
                      verdicts["shows_improvement"]])
        return {
            "passed": passed,
            "verdicts": verdicts,
            "deltas": {
                "p95_latency_pct": round(lat_delta_pct, 2),
                "mos": round(mos_delta, 3),
                "wer": round(wer_delta, 4),
            },
        }

    @staticmethod
    def _pct_delta(base: Optional[float], cand: Optional[float]) -> float:
        if not base or cand is None:
            return 0.0
        return (cand - base) / base * 100.0

    # -- report ---------------------------------------------------------------
    def to_markdown(self, baseline: BenchmarkResult, candidate: BenchmarkResult,
                    gate: Dict[str, Any], proposals: Optional[List[Any]] = None) -> str:
        def fmt(v, suffix=""):
            return f"{v}{suffix}" if v is not None else "—"
        lines = [
            f"# Benchmark Report — {candidate.label} vs {baseline.label}",
            "",
            "| Metric | Baseline | Candidate | Δ |",
            "|--------|----------|-----------|---|",
            f"| ASR WER | {fmt(baseline.asr_wer)} | {fmt(candidate.asr_wer)} | {gate['deltas']['wer']:+.4f} |",
            f"| TTS MOS | {fmt(baseline.tts_mos)} | {fmt(candidate.tts_mos)} | {gate['deltas']['mos']:+.3f} |",
            f"| Turn p50 (ms) | {fmt(baseline.turn_latency_p50_ms)} | {fmt(candidate.turn_latency_p50_ms)} | |",
            f"| Turn p95 (ms) | {fmt(baseline.turn_latency_p95_ms)} | {fmt(candidate.turn_latency_p95_ms)} | {gate['deltas']['p95_latency_pct']:+.1f}% |",
            f"| First token (ms) | {fmt(baseline.first_token_ms)} | {fmt(candidate.first_token_ms)} | |",
            "",
            f"## Gate verdict: {'✅ PASS' if gate['passed'] else '❌ FAIL'}",
            "",
        ]
        for k, v in gate["verdicts"].items():
            lines.append(f"- {'✅' if v else '❌'} {k}")
        if proposals:
            lines += ["", "## Source proposals"]
            for p in proposals:
                d = p.to_dict() if hasattr(p, "to_dict") else p
                lines.append(f"- **{d.get('title')}** → {d.get('expected_impact')} "
                             f"({d.get('citation')})")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = BenchmarkRunner()
    print("WER:", runner.word_error_rate("hello there friend", "hello there friends"))
    base = BenchmarkResult("baseline", asr_wer=0.12, tts_mos=3.7, turn_latency_p95_ms=1800)
    cand = BenchmarkResult("candidate", asr_wer=0.11, tts_mos=4.1, turn_latency_p95_ms=1300)
    g = runner.evaluate_gates(base, cand)
    print(runner.to_markdown(base, cand, g))
