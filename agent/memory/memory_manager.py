"""Persistent memory for open-llm-vtuber-enhanced.

SQLite (WAL mode) store covering five concerns:
  * conversation turns       -> long-term persona memory & context recall
  * benchmark runs           -> before/after latency / WER / MOS comparisons
  * llm cost log             -> per-session spend accounting
  * knowledge hashes         -> paper dedup for the research crawler
  * human MOS ratings        -> human naturalness ratings to calibrate the
                                objective MOS proxy (Phase 7)

The conversation table is what gives the VTuber its "long-term memory": the
orchestrator recalls the most recent / most relevant turns for a given user so
the persona stays consistent across sessions.
"""

from __future__ import annotations

import os
import json
import time
import sqlite3
import threading
import datetime as dt
from typing import List, Dict, Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    user_id       TEXT DEFAULT 'default',
    role          TEXT NOT NULL,            -- 'user' | 'assistant'
    text          TEXT NOT NULL,
    emotion       TEXT,                     -- dominant emotion label
    asr_latency_ms   REAL DEFAULT 0,
    llm_latency_ms   REAL DEFAULT 0,
    tts_latency_ms   REAL DEFAULT 0,
    turn_latency_ms  REAL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_user ON conversation_turns(user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id, id);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT NOT NULL,
    asr_wer       REAL,
    tts_mos       REAL,
    turn_latency_p50_ms REAL,
    turn_latency_p95_ms REAL,
    first_token_ms      REAL,
    notes         TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cost_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT,
    provider      TEXT,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    latency_ms    REAL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_hashes (
    url_hash      TEXT PRIMARY KEY,
    title         TEXT,
    url           TEXT,
    added_at      TEXT NOT NULL
);

-- Phase 7: human MOS ratings to calibrate the objective MOS proxy.
-- A rater listens to a synthesized clip (audio_path) and scores naturalness
-- on the standard 1-5 MOS scale. The rater can be a human id ("human") or,
-- later, an automated reference metric used for cross-validation.
CREATE TABLE IF NOT EXISTS human_mos_ratings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_path    TEXT NOT NULL,
    text          TEXT,                      -- prompt that produced the clip
    emotion       TEXT,
    rater_id      TEXT NOT NULL DEFAULT 'human',
    rating        REAL NOT NULL,             -- 1.0 .. 5.0
    proxy_mos     REAL,                      -- objective proxy at rating time
    notes         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mos_audio ON human_mos_ratings(audio_path);
CREATE INDEX IF NOT EXISTS idx_mos_rater ON human_mos_ratings(rater_id, id DESC);
"""


class MemoryManager:
    def __init__(self, db_path: str = "./data/vtuber_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @staticmethod
    def _now() -> str:
        return dt.datetime.now().isoformat(timespec="seconds")

    # -- conversation ---------------------------------------------------------
    def save_turn(self, session_id: str, role: str, text: str,
                  user_id: str = "default", emotion: Optional[str] = None,
                  latencies: Optional[Dict[str, float]] = None) -> int:
        latencies = latencies or {}
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO conversation_turns
                   (session_id, user_id, role, text, emotion,
                    asr_latency_ms, llm_latency_ms, tts_latency_ms, turn_latency_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (session_id, user_id, role, text, emotion,
                 latencies.get("asr", 0.0), latencies.get("llm", 0.0),
                 latencies.get("tts", 0.0), latencies.get("turn", 0.0), self._now()))
            self._conn.commit()
            return cur.lastrowid

    def recent_turns(self, user_id: str = "default", limit: int = 12) -> List[Dict[str, Any]]:
        """Most recent turns (chronological) for context window assembly."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT role, text, emotion, created_at FROM conversation_turns
                   WHERE user_id=? ORDER BY id DESC LIMIT ?""",
                (user_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search_memory(self, user_id: str, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Naive keyword recall over long-term memory (semantic recall is done
        by the orchestrator via embeddings; this is the durable fallback)."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT role, text, emotion, created_at FROM conversation_turns
                   WHERE user_id=? AND text LIKE ? ORDER BY id DESC LIMIT ?""",
                (user_id, f"%{keyword}%", limit)).fetchall()
        return [dict(r) for r in rows]

    def all_user_texts(self, user_id: str = "default", limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, role, text FROM conversation_turns
                   WHERE user_id=? ORDER BY id DESC LIMIT ?""",
                (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- benchmarks -----------------------------------------------------------
    def save_benchmark(self, label: str, metrics: Dict[str, float],
                       notes: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO benchmark_runs
                   (label, asr_wer, tts_mos, turn_latency_p50_ms,
                    turn_latency_p95_ms, first_token_ms, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (label, metrics.get("asr_wer"), metrics.get("tts_mos"),
                 metrics.get("turn_latency_p50_ms"), metrics.get("turn_latency_p95_ms"),
                 metrics.get("first_token_ms"), notes, self._now()))
            self._conn.commit()
            return cur.lastrowid

    def benchmark_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def latest_benchmark(self, label: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            if label:
                row = self._conn.execute(
                    "SELECT * FROM benchmark_runs WHERE label=? ORDER BY id DESC LIMIT 1",
                    (label,)).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    # -- cost -----------------------------------------------------------------
    def log_llm_cost(self, record: Dict[str, Any]):
        with self._lock:
            self._conn.execute(
                """INSERT INTO llm_cost_log
                   (session_id, provider, model, input_tokens, output_tokens,
                    cost_usd, latency_ms, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (record.get("session_id"), record.get("provider"), record.get("model"),
                 record.get("input_tokens", 0), record.get("output_tokens", 0),
                 record.get("cost_usd", 0.0), record.get("latency_ms", 0.0), self._now()))
            self._conn.commit()

    def cost_summary(self, days: int = 30) -> Dict[str, Any]:
        cutoff = (dt.datetime.now() - dt.timedelta(days=days)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                """SELECT provider, COUNT(*) n, SUM(cost_usd) total,
                          SUM(input_tokens) intok, SUM(output_tokens) outtok
                   FROM llm_cost_log WHERE created_at >= ? GROUP BY provider""",
                (cutoff,)).fetchall()
        by_provider = {r["provider"]: {
            "calls": r["n"], "cost_usd": round(r["total"] or 0.0, 4),
            "input_tokens": r["intok"] or 0, "output_tokens": r["outtok"] or 0}
            for r in rows}
        return {"window_days": days, "by_provider": by_provider,
                "total_usd": round(sum(v["cost_usd"] for v in by_provider.values()), 4)}

    # -- knowledge dedup ------------------------------------------------------
    def is_known_paper(self, url_hash: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM knowledge_hashes WHERE url_hash=?", (url_hash,)).fetchone()
        return row is not None

    def mark_paper_known(self, url_hash: str, title: str, url: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO knowledge_hashes (url_hash, title, url, added_at) VALUES (?,?,?,?)",
                (url_hash, title, url, self._now()))
            self._conn.commit()

    # -- human MOS ratings (Phase 7) -----------------------------------------
    def save_mos_rating(self, audio_path: str, rating: float,
                        text: Optional[str] = None, emotion: Optional[str] = None,
                        rater_id: str = "human",
                        proxy_mos: Optional[float] = None,
                        notes: str = "") -> int:
        if not (1.0 <= float(rating) <= 5.0):
            raise ValueError("MOS rating must be between 1.0 and 5.0")
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO human_mos_ratings
                   (audio_path, text, emotion, rater_id, rating, proxy_mos, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (audio_path, text, emotion, rater_id, float(rating),
                 proxy_mos, notes, self._now()))
            self._conn.commit()
            return cur.lastrowid

    def mos_ratings(self, rater_id: Optional[str] = None,
                    limit: int = 1000) -> List[Dict[str, Any]]:
        with self._lock:
            if rater_id:
                rows = self._conn.execute(
                    """SELECT * FROM human_mos_ratings WHERE rater_id=?
                       ORDER BY id DESC LIMIT ?""",
                    (rater_id, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM human_mos_ratings ORDER BY id DESC LIMIT ?",
                    (limit,)).fetchall()
        return [dict(r) for r in rows]

    def mos_summary(self) -> Dict[str, Any]:
        """Aggregate human MOS ratings and pair them with the proxy MOS so the
        calibration routine can compute the offset between the two."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) n, AVG(rating) avg_human, AVG(proxy_mos) avg_proxy,
                          MIN(rating) min_human, MAX(rating) max_human
                   FROM human_mos_ratings""").fetchone()
            by_emotion = self._conn.execute(
                """SELECT emotion, COUNT(*) n, AVG(rating) avg_human, AVG(proxy_mos) avg_proxy
                   FROM human_mos_ratings GROUP BY emotion""").fetchall()
        n = row["n"] if row else 0
        return {
            "n": n,
            "avg_human_mos": round(row["avg_human"], 3) if n and row["avg_human"] else None,
            "avg_proxy_mos": round(row["avg_proxy"], 3) if n and row["avg_proxy"] else None,
            "min_human": row["min_human"] if n else None,
            "max_human": row["max_human"] if n else None,
            "by_emotion": [
                {"emotion": r["emotion"], "n": r["n"],
                 "avg_human": round(r["avg_human"], 3) if r["avg_human"] else None,
                 "avg_proxy": round(r["avg_proxy"], 3) if r["avg_proxy"] else None}
                for r in by_emotion],
        }

    # -- stats ----------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            turns = self._conn.execute("SELECT COUNT(*) c FROM conversation_turns").fetchone()["c"]
            benches = self._conn.execute("SELECT COUNT(*) c FROM benchmark_runs").fetchone()["c"]
            papers = self._conn.execute("SELECT COUNT(*) c FROM knowledge_hashes").fetchone()["c"]
            mos = self._conn.execute("SELECT COUNT(*) c FROM human_mos_ratings").fetchone()["c"]
        return {"conversation_turns": turns, "benchmark_runs": benches,
                "known_papers": papers, "human_mos_ratings": mos,
                "cost": self.cost_summary()}

    def close(self):
        with self._lock:
            self._conn.close()


if __name__ == "__main__":
    mm = MemoryManager("./data/_test_mem.db")
    sid = "s1"
    mm.save_turn(sid, "user", "hi there", emotion="joy",
                 latencies={"asr": 120, "llm": 300, "tts": 200, "turn": 700})
    mm.save_turn(sid, "assistant", "Hello! So happy to see you~", emotion="joy")
    print("recent:", mm.recent_turns())
    mm.save_mos_rating("/tmp/clip.wav", 4.3, text="hello", emotion="joy", proxy_mos=4.0)
    print("mos summary:", mm.mos_summary())
    print("stats:", mm.get_stats())
    mm.close()
