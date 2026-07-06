"""Calibrate the objective MOS proxy against collected human MOS ratings.

Phase 7 deliverable. Until enough human ratings exist the pipeline uses an
objective MOS proxy (benchmark_runner.mos_proxy) so the quality gate is runnable
in CI. Once human ratings accumulate, this script fits a linear calibration
(proxy -> human) and emits calibration constants the orchestrator can apply to
bring the proxy in line with perceived naturalness.

Outputs ``data/mos_calibration.json`` with:
  * slope / intercept of the least-squares fit
  * mean bias (proxy - human) and per-emotion biases
  * a recommended calibrated_mos(proxy) function the agent can apply
  * sample count + fit quality (R^2)

When fewer than ``--min-samples`` ratings are available the script reports
"insufficient data" and writes a neutral calibration (slope=1, intercept=0),
so downstream code can always load a calibration file.

Usage:
    python -m scripts.calibrate_mos
    python -m scripts.calibrate_mos --min-samples 20 --rater-id human
"""

from __future__ import annotations

import os
import json
import argparse
import logging
import datetime as dt
from typing import List, Dict, Any, Tuple

from scripts._common import build_orchestrator, ROOT

logger = logging.getLogger("vtuber.calibrate_mos")

CALIBRATION_PATH = os.path.join(ROOT, "data", "mos_calibration.json")


def _least_squares(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for y = slope*x + intercept."""
    n = len(xs)
    if n < 2:
        return 1.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x == 0:
        return 1.0, 0.0, 0.0
    slope = num / den_x
    intercept = mean_y - slope * mean_x
    r2 = (num * num) / (den_x * den_y) if den_y > 0 else 0.0
    return slope, intercept, r2


def run(rater_id: str, min_samples: int,
        out_path: str = CALIBRATION_PATH) -> Dict[str, Any]:
    orch = build_orchestrator()
    mem = orch.memory()
    ratings = mem.mos_ratings(rater_id=rater_id)
    pairs = [(float(r["proxy_mos"]), float(r["rating"]))
             for r in ratings
             if r.get("proxy_mos") is not None and r.get("rating") is not None]
    summary = mem.mos_summary()

    if len(pairs) < min_samples:
        calibration = {
            "status": "insufficient_data",
            "min_samples_required": min_samples,
            "available_samples": len(pairs),
            "slope": 1.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "calibrated_mos": "proxy_mos * 1.0 + 0.0",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "rater_id": rater_id,
        }
        _write(out_path, calibration)
        return calibration

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    slope, intercept, r2 = _least_squares(xs, ys)
    mean_bias = round(sum(px - hy for px, hy in zip(xs, ys)) / len(pairs), 4)

    # per-emotion bias
    by_emotion: Dict[str, List[float]] = {}
    for r in ratings:
        if r.get("proxy_mos") is None or r.get("rating") is None:
            continue
        emo = r.get("emotion") or "neutral"
        by_emotion.setdefault(emo, []).append(float(r["proxy_mos"]) - float(r["rating"]))
    emotion_bias = {e: round(sum(v) / len(v), 4) for e, v in by_emotion.items()}

    calibration = {
        "status": "calibrated",
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r2, 4),
        "mean_bias": mean_bias,
        "per_emotion_bias": emotion_bias,
        "calibrated_mos": f"proxy_mos * {round(slope, 4)} + {round(intercept, 4)}",
        "samples": len(pairs),
        "rater_id": rater_id,
        "summary": summary,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _write(out_path, calibration)
    return calibration


def _write(path: str, calibration: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=2, ensure_ascii=False)
    logger.info("MOS calibration written to %s (status=%s)",
                path, calibration.get("status"))


def load_calibration(path: str = CALIBRATION_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"slope": 1.0, "intercept": 0.0, "status": "default"}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def calibrated_mos(proxy: float, path: str = CALIBRATION_PATH) -> float:
    cal = load_calibration(path)
    return round(float(proxy) * float(cal.get("slope", 1.0))
                 + float(cal.get("intercept", 0.0)), 3)


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Calibrate the MOS proxy vs human ratings.")
    ap.add_argument("--rater-id", default="human")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--out", default=CALIBRATION_PATH)
    args = ap.parse_args()
    cal = run(args.rater_id, args.min_samples, args.out)
    print(json.dumps(cal, indent=2))
    print(f"calibration file: {args.out}")


if __name__ == "__main__":
    main()
