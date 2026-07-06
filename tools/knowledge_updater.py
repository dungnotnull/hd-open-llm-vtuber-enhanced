"""Research-paper crawler for open-llm-vtuber-enhanced.

This is the "research agent" called for in idea.txt: it continuously ingests
the latest conversational-AI / speech / HCI papers and appends the highest
scored entries to ``SECOND-KNOWLEDGE-BRAIN.md``. The improvement_proposer
module reads that file to ground its optimization proposals, so the longer this
crawler runs, the better the agent's upgrade suggestions become.

Pipeline (per CLAUDE.md universal component #1):
  1. Fetch from ArXiv (cs.CL, cs.SD, cs.HC, eess.AS) + Semantic Scholar.
  2. Parse -> title, authors, date, URL, abstract.
  3. Score by recency (<=90 days highest) x domain-keyword relevance.
  4. Deduplicate via SHA256 of the canonical URL (checked against memory).
  5. Append top-N to SECOND-KNOWLEDGE-BRAIN.md with ISO date stamp.
  6. Notify (return a summary dict).

Networking uses only the stdlib (urllib) so the crawler runs with no extra
deps; APScheduler is used for the weekly/daily schedule when available.
"""

from __future__ import annotations

import os
import re
import time
import hashlib
import logging
import datetime as dt
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import xml.etree.ElementTree as ET

logger = logging.getLogger("vtuber.knowledge_updater")

ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Domain queries tuned for the VTuber speech/dialogue/HCI stack.
ARXIV_QUERIES = [
    "cat:cs.CL AND (streaming ASR OR speech recognition latency)",
    "cat:cs.SD AND (text to speech OR voice cloning OR neural vocoder)",
    "cat:eess.AS AND (real-time speech OR low latency synthesis)",
    "cat:cs.HC AND (conversational agent OR virtual avatar OR live2d)",
    "cat:cs.CL AND (dialogue persona OR long-term memory conversational)",
]

S2_QUERIES = [
    "streaming speech recognition low latency",
    "zero-shot voice cloning text-to-speech",
    "emotion-aware talking avatar animation",
    "persona-grounded dialogue large language model",
    "real-time conversational AI turn-taking latency",
]

DOMAIN_KEYWORDS = [
    "asr", "speech recognition", "tts", "text-to-speech", "voice cloning",
    "vocoder", "diarization", "latency", "streaming", "real-time", "dialogue",
    "persona", "avatar", "lip-sync", "viseme", "emotion", "turn-taking",
    "conversational", "whisper", "xtts", "live2d", "vtuber", "duplex",
]

RECENCY_WINDOW_DAYS = 90


@dataclass
class PaperEntry:
    title: str
    authors: str
    date: str           # ISO YYYY-MM-DD
    url: str
    abstract: str
    source: str
    score: float = 0.0
    key_finding: str = ""

    @property
    def url_hash(self) -> str:
        return hashlib.sha256(self.url.strip().lower().encode()).hexdigest()


class KnowledgeUpdater:
    def __init__(self, brain_path: str, memory=None, top_n: int = 8,
                 summarizer=None):
        self.brain_path = brain_path
        self.memory = memory            # MemoryManager for dedup (optional)
        self.top_n = top_n
        self.summarizer = summarizer    # callable(text)->str (optional, BART)

    # -- fetch ----------------------------------------------------------------
    def _http_get(self, url: str, timeout: int = 30) -> Optional[bytes]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vtuber-research-agent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTTP GET failed (%s): %s", url, exc)
            return None

    def fetch_arxiv(self, query: str, max_results: int = 10) -> List[PaperEntry]:
        params = urllib.parse.urlencode({
            "search_query": query, "start": 0, "max_results": max_results,
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        raw = self._http_get(f"{ARXIV_API}?{params}")
        if not raw:
            return []
        return self._parse_arxiv_xml(raw)

    def _parse_arxiv_xml(self, raw: bytes) -> List[PaperEntry]:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: List[PaperEntry] = []
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            logger.warning("arxiv XML parse error: %s", exc)
            return out
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("a:published", default="", namespaces=ns) or "")[:10]
            link = entry.findtext("a:id", default="", namespaces=ns) or ""
            authors = ", ".join(
                a.findtext("a:name", default="", namespaces=ns)
                for a in entry.findall("a:author", ns))[:200]
            out.append(PaperEntry(
                title=re.sub(r"\s+", " ", title), authors=authors,
                date=published or self._today(), url=link.strip(),
                abstract=re.sub(r"\s+", " ", summary), source="arxiv"))
        return out

    def fetch_semantic_scholar(self, query: str, max_results: int = 10) -> List[PaperEntry]:
        params = urllib.parse.urlencode({
            "query": query, "limit": max_results,
            "fields": "title,abstract,year,url,authors,publicationDate",
        })
        raw = self._http_get(f"{S2_API}?{params}")
        if not raw:
            return []
        import json
        try:
            data = json.loads(raw.decode())
        except Exception:  # noqa: BLE001
            return []
        out = []
        for p in data.get("data", []):
            if not p.get("title"):
                continue
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or []))[:200]
            date = p.get("publicationDate") or (f"{p.get('year')}-01-01" if p.get("year") else self._today())
            out.append(PaperEntry(
                title=p["title"].strip(), authors=authors, date=date[:10],
                url=(p.get("url") or "").strip(),
                abstract=(p.get("abstract") or "").strip(), source="semantic_scholar"))
        return out

    # -- score ----------------------------------------------------------------
    def _score(self, paper: PaperEntry) -> float:
        # recency component
        try:
            pub = dt.date.fromisoformat(paper.date)
            age_days = (dt.date.today() - pub).days
        except Exception:  # noqa: BLE001
            age_days = 9999
        recency = max(0.0, 1.0 - age_days / max(1, RECENCY_WINDOW_DAYS)) if age_days <= RECENCY_WINDOW_DAYS \
            else max(0.0, 0.3 - (age_days - RECENCY_WINDOW_DAYS) / 3650.0)
        # relevance component
        blob = f"{paper.title} {paper.abstract}".lower()
        hits = sum(1 for kw in DOMAIN_KEYWORDS if kw in blob)
        relevance = min(1.0, hits / 6.0)
        return round(0.6 * recency + 0.4 * relevance, 4)

    # -- dedup ----------------------------------------------------------------
    def _is_known(self, paper: PaperEntry) -> bool:
        if self.memory is not None:
            try:
                return self.memory.is_known_paper(paper.url_hash)
            except Exception:  # noqa: BLE001
                pass
        return False

    def _mark_known(self, paper: PaperEntry):
        if self.memory is not None:
            try:
                self.memory.mark_paper_known(paper.url_hash, paper.title, paper.url)
            except Exception:  # noqa: BLE001
                pass

    # -- run ------------------------------------------------------------------
    def run_once(self) -> Dict[str, Any]:
        """Execute the full crawl pipeline once. Returns a summary dict."""
        collected: List[PaperEntry] = []
        for q in ARXIV_QUERIES:
            collected.extend(self.fetch_arxiv(q))
            time.sleep(1)  # be polite to the arXiv API
        for q in S2_QUERIES:
            collected.extend(self.fetch_semantic_scholar(q))
            time.sleep(1)

        # dedup within this batch by url_hash
        seen, unique = set(), []
        for p in collected:
            if not p.url or p.url_hash in seen:
                continue
            seen.add(p.url_hash)
            unique.append(p)

        # filter already-known and score
        fresh = [p for p in unique if not self._is_known(p)]
        for p in fresh:
            p.score = self._score(p)
            if self.summarizer and p.abstract:
                try:
                    p.key_finding = self.summarizer(p.abstract)
                except Exception:  # noqa: BLE001
                    p.key_finding = p.abstract[:160]
            else:
                p.key_finding = p.abstract[:160]

        fresh.sort(key=lambda x: x.score, reverse=True)
        top = fresh[: self.top_n]
        if top:
            self._append_to_brain(top)
            for p in top:
                self._mark_known(p)

        summary = {
            "fetched": len(collected),
            "unique": len(unique),
            "new": len(top),
            "next_run": "weekly (Sunday 02:00 local)",
            "titles": [p.title for p in top],
        }
        logger.info("knowledge update: %d fetched, %d new appended", len(collected), len(top))
        return summary

    def _append_to_brain(self, papers: List[PaperEntry]):
        os.makedirs(os.path.dirname(os.path.abspath(self.brain_path)) or ".", exist_ok=True)
        stamp = self._today()
        lines = [f"\n### Auto-update {stamp}\n",
                 "| Title | Authors | Date | Source | Score | Key Finding | Link |",
                 "|-------|---------|------|--------|-------|-------------|------|"]
        for p in papers:
            title = p.title.replace("|", "/")
            finding = (p.key_finding or "").replace("|", "/").replace("\n", " ")[:200]
            lines.append(f"| {title} | {p.authors[:60]} | {p.date} | {p.source} "
                         f"| {p.score:.3f} | {finding} | {p.url} |")
        with open(self.brain_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    @staticmethod
    def _today() -> str:
        return dt.date.today().isoformat()

    # -- scheduling -----------------------------------------------------------
    def start_scheduler(self, cron: str = "weekly"):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except Exception as exc:  # noqa: BLE001
            logger.warning("APScheduler unavailable (%s); run_once must be called manually", exc)
            return None
        sched = BackgroundScheduler()
        if cron == "daily":
            sched.add_job(self.run_once, "cron", hour=6, minute=0, id="kb_daily")
        else:
            sched.add_job(self.run_once, "cron", day_of_week="sun", hour=2, minute=0, id="kb_weekly")
        sched.start()
        logger.info("knowledge scheduler started (%s)", cron)
        return sched


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    updater = KnowledgeUpdater(os.path.join(here, "SECOND-KNOWLEDGE-BRAIN.md"))
    print(updater.run_once())
