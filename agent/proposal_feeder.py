"""Proposal feeder: export research proposals to ``academic-research-enhanced``.

Phase 7 deliverable. The improvement_proposer generates cited upgrade proposals
for the VTuber pipeline; this feeder serializes them into the shared
inter-project proposal feed consumed by ``academic-research-enhanced`` (folder
18) for deeper cross-domain synthesis.

Feed contract (``data/proposal_feed/``):
  * ``proposals_<UTC date>.json``  - one file per generation run (snapshot)
  * ``feed.jsonl``                  - append-only, dedup-by-citation cumulative
                                      stream that downstream agents tail
  * ``latest.json``                 - pointer to the most recent snapshot

Each feed record is a self-contained proposal with its citation, the metrics it
was based on, the source pipeline identity, and an ISO timestamp, so a
downstream synthesis agent can de-duplicate, rank, and combine proposals across
projects without needing access to this repo's internals.
"""

from __future__ import annotations

import os
import json
import logging
import datetime as dt
from typing import List, Dict, Any, Optional

logger = logging.getLogger("vtuber.proposal_feeder")

FEED_DIRNAME = "proposal_feed"


class ProposalFeeder:
    def __init__(self, base_dir: str, source_pipeline: str = "open-llm-vtuber-enhanced"):
        self.base_dir = base_dir
        self.feed_dir = os.path.join(base_dir, "data", FEED_DIRNAME)
        self.source_pipeline = source_pipeline
        os.makedirs(self.feed_dir, exist_ok=True)
        self.feed_path = os.path.join(self.feed_dir, "feed.jsonl")

    # -- paths ---------------------------------------------------------------
    def snapshot_path(self, when: Optional[dt.datetime] = None) -> str:
        when = when or dt.datetime.now(dt.timezone.utc)
        return os.path.join(self.feed_dir,
                             f"proposals_{when.strftime('%Y%m%d')}.json")

    @property
    def latest_path(self) -> str:
        return os.path.join(self.feed_dir, "latest.json")

    # -- write ----------------------------------------------------------------
    def feed(self, proposals: List[Dict[str, Any]],
             based_on: Dict[str, Any]) -> Dict[str, Any]:
        """Write a snapshot + append the cumulative feed. Returns a summary."""
        if not proposals:
            logger.info("no proposals to feed; skipping feed write")
            return {"written": 0, "snapshot": None, "feed": self.feed_path}

        now = dt.datetime.now(dt.timezone.utc)
        records: List[Dict[str, Any]] = []
        existing = self._existing_citations()
        appended = 0
        for p in proposals:
            citation = (p.get("citation") or "").strip()
            if not citation or citation in existing:
                continue
            existing.add(citation)
            records.append({
                "id": f"{self.source_pipeline}:{_slug(p.get('title', 'untitled'))}:{citation}",
                "source_pipeline": self.source_pipeline,
                "generated_at": now.isoformat(timespec="seconds"),
                "based_on_metrics": based_on,
                "title": p.get("title", ""),
                "target_metric": p.get("target_metric", "overall"),
                "change": p.get("change", ""),
                "rationale": p.get("rationale", ""),
                "expected_impact": p.get("expected_impact", ""),
                "risk": p.get("risk", "medium"),
                "citation": citation,
            })
            appended += 1

        snapshot_path = self.snapshot_path(now)
        snapshot = {
            "source_pipeline": self.source_pipeline,
            "generated_at": now.isoformat(timespec="seconds"),
            "based_on_metrics": based_on,
            "count": len(records),
            "proposals": records,
        }
        # Only update snapshot and latest pointer if we actually added new proposals
        if records:
            with open(snapshot_path, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2, ensure_ascii=False)
            with open(self.latest_path, "w", encoding="utf-8") as fh:
                json.dump({"snapshot": snapshot_path, "generated_at": snapshot["generated_at"],
                           "count": len(records)}, fh, indent=2, ensure_ascii=False)
            with open(self.feed_path, "a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("fed %d new proposals (deduped %d) to %s",
                    appended, len(proposals) - appended, self.feed_path)
        return {"written": appended, "snapshot": snapshot_path if records else None,
                "feed": self.feed_path, "latest": self.latest_path if records else None}

    def _existing_citations(self) -> set:
        if not os.path.exists(self.feed_path):
            return set()
        seen = set()
        with open(self.feed_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("citation"):
                        seen.add(rec["citation"])
                except json.JSONDecodeError:
                    continue
        return seen

    # -- read -----------------------------------------------------------------
    def read_feed(self, limit: int = 1000) -> List[Dict[str, Any]]:
        if not os.path.exists(self.feed_path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.feed_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(out) >= limit:
                    break
        return out

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.latest_path):
            return None
        with open(self.latest_path, "r", encoding="utf-8") as fh:
            pointer = json.load(fh)
        snap = pointer.get("snapshot")
        if snap and os.path.exists(snap):
            with open(snap, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None


def _slug(text: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "proposal"


if __name__ == "__main__":
    import tempfile
    feeder = ProposalFeeder(tempfile.mkdtemp())
    feeder.feed([{"title": "Test proposal", "citation": "https://arxiv.org/abs/1",
                  "change": "x", "rationale": "y", "expected_impact": "+1",
                  "risk": "low", "target_metric": "latency"}],
                {"asr_wer": 0.1})
    print(feeder.read_feed())
