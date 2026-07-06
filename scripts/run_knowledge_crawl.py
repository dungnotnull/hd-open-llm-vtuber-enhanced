"""Run the research-paper crawl once (Phase 5 ops entry point).

Designed to be invoked by an external scheduler (cron / systemd timer /
GitHub Actions scheduled workflow) when the agent is not running a long-lived
server, or as a one-shot operator command. When the FastAPI server is running
with its built-in daily scheduler (see agent.main), this script is redundant.

Exit codes:
  0  crawl completed (possibly zero new papers; dedup is expected)
  1  crawl raised an exception (logged, never crashes the host)

Usage:
    python -m scripts.run_knowledge_crawl
    python -m scripts.run_knowledge_crawl --top-n 12
    python -m scripts.run_knowledge_crawl --schedule daily   # start in-process scheduler
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from scripts._common import build_orchestrator, ROOT

logger = logging.getLogger("vtuber.run_knowledge_crawl")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run the research-paper crawl once "
                                              "(or start the in-process scheduler).")
    ap.add_argument("--top-n", type=int, default=None,
                    help="override max papers appended per run")
    ap.add_argument("--schedule", choices=("daily", "weekly"), default=None,
                    help="if set, start the in-process APScheduler and block "
                         "(use for production deployments without an external cron)")
    args = ap.parse_args()

    orch = build_orchestrator()
    if args.top_n is not None:
        orch.knowledge().top_n = args.top_n

    if args.schedule:
        sched = orch.start_scheduler(cron=args.schedule)
        if sched is None:
            print("APScheduler unavailable; cannot run in-process schedule. "
                  "Use an external cron invoking this script without --schedule.")
            sys.exit(1)
        print(f"in-process {args.schedule} scheduler started; press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            sched.shutdown(wait=False)
            sys.exit(0)

    try:
        summary = orch.update_knowledge()
    except Exception as exc:  # noqa: BLE001 - crawl must never crash the host
        logger.exception("knowledge crawl failed: %s", exc)
        print(json.dumps({"error": str(exc), "new": 0}))
        sys.exit(1)

    summary["brain_path"] = orch.brain_path
    print(json.dumps(summary, indent=2))
    # dedup success gate: a clean re-run reports new == 0
    sys.exit(0)


if __name__ == "__main__":
    main()
