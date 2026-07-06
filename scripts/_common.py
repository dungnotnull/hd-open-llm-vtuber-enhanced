"""Shared helpers for the operational / validation scripts.

Keeps each script small while sharing config loading, project-root resolution
and Markdown-report writing. Scripts run with no ML deps (the agent degrades
gracefully), but they are written to be fully correct when models are present
so they can validate the SOTA speech stack in production.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_config() -> Dict[str, Any]:
    cfg_path = os.path.join(ROOT, "config", "agent_config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logging.warning("config load failed (%s); using defaults", exc)
        return {}


def build_orchestrator():
    from agent.orchestrator import VTuberOrchestrator
    return VTuberOrchestrator(config=load_config(), base_dir=ROOT)


def write_report(path: str, title: str, lines: List[str]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    body = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n{body}\n")
    return path


def load_manifest(path: str) -> List[Dict[str, str]]:
    """Load a validation manifest.

    Supported formats:
      * JSONL  - one ``{"audio": "...", "reference": "..."}`` per line
      * CSV     - header with ``audio,reference`` columns
      * JSON    - a list of the same objects

    Audio paths are resolved relative to the manifest directory when they are
    not absolute.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest not found: {path}")
    base = os.path.dirname(os.path.abspath(path))
    rows: List[Dict[str, str]] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(_resolve_row(json.loads(line), base))
    elif path.lower().endswith(".csv"):
        import csv
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(_resolve_row(row, base))
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for row in data:
            rows.append(_resolve_row(row, base))
    if not rows:
        raise ValueError(f"manifest {path} contained no rows")
    return rows


def _resolve_row(row: Dict[str, Any], base: str) -> Dict[str, str]:
    audio = row.get("audio") or row.get("path") or row.get("file") or ""
    if audio and not os.path.isabs(audio):
        audio = os.path.join(base, audio)
    return {
        "audio": audio,
        "reference": (row.get("reference") or row.get("text") or "").strip(),
        "emotion": (row.get("emotion") or "").strip() or None,
        "language": (row.get("language") or "").strip() or None,
    }
