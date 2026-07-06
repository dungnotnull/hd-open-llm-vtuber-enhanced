# CLAUDE.md — open-llm-vtuber-enhanced

**Tagline:** Real-time interactive AI VTuber & voice companion with a built-in
research agent that continuously proposes cited pipeline optimizations.
**Cluster:** A — Real-time Multimedia & Speech AI Agents
**Upstream fork:** [Open-LLM-VTuber/Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
**Current build phase:** Phase 0–6 deliverables authored (greenfield AI layer + sidecar over the fork).
**Port:** 8001

---

## Problem Statement
Open-LLM-VTuber gives you a working `ASR → LLM → TTS → Live2D` voice-companion
pipeline, but it is static: it does not measure its own latency/quality and it
does not know about newer, faster, or more natural models as research moves. This
agent forks that pipeline and adds (1) a SOTA-model speech stack, (2) an
emotion-aware avatar, persona-grounded dialogue with long-term memory, and
(3) an autonomous **research agent** that crawls the latest speech/dialogue/HCI
papers daily, benchmarks the current pipeline, and synthesizes *cited* upgrade
proposals — accepting only changes that improve latency or quality without
regressing the other. The longer it runs, the better its proposals get.

## Agent Architecture (turn loop + optimization loop)
1. **Ingest** (`modules/stream_ingestor.py`) — capture mic / file / text, VAD &
   quality gate.
2. **Understand** (`modules/speech_processor.py`) — Whisper-large-v3 ASR +
   optional pyannote diarization + emotion classification.
3. **Recall + Reason** (`orchestrator.py` + `memory/memory_manager.py`) — pull
   long-term memory, generate a persona-grounded reply via the LLM client.
4. **Synthesize** (`modules/media_synthesizer.py`) — XTTS-v2 voice + emotion →
   Live2D expression/motion + viseme lip-sync timeline.
5. **Persist** — store the turn with per-stage latency.
6. **Research loop** (`modules/improvement_proposer.py` + `tools/knowledge_updater.py`)
   — crawl papers → propose cited optimizations.
7. **Verify** (`modules/benchmark_runner.py`) — measure WER / MOS / latency
   before vs after; gate the change.

## Modules (`agent/modules/`)
- `stream_ingestor.py` — real-time mic/file/text ingestion + audio quality gate.
- `speech_processor.py` — Whisper ASR + diarization + emotion on the input side.
- `media_synthesizer.py` — XTTS-v2 TTS + emotion→Live2D expression + lip-sync.
- `improvement_proposer.py` — research-grounded, cited optimization proposals.
- `benchmark_runner.py` — WER/MOS/latency measurement + before/after quality gate.

## Tools (`agent/` + `tools/`)
- `agent/orchestrator.py` — turn loop + optimization loop, lazy module singletons.
- `agent/main.py` — Click CLI + FastAPI server.
- `agent/memory/memory_manager.py` — SQLite (WAL) conversations / benchmarks / cost / paper hashes.

## HuggingFace Models
| Task | Model | Why |
|------|-------|-----|
| Streaming ASR | `openai/whisper-large-v3` | SOTA multilingual WER; word timestamps for lip-sync. |
| Diarization | `pyannote/speaker-diarization-3.1` | Best open diarization for multi-speaker rooms. |
| Zero-shot TTS | `coqui/XTTS-v2` | 6s reference voice cloning, expressive, multilingual. |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | Fast 7-class emotion → avatar expression. |
| Embeddings | `BAAI/bge-large-en-v1.5` | Strong retrieval for paper grounding. |
| Reranking | `BAAI/bge-reranker-large` | Precision rerank of retrieved papers. |
| Summarization | `facebook/bart-large-cnn` | Compress abstracts to fit LLM context. |

All models lazy-load and degrade to deterministic fallbacks (graceful degradation).

## LLM API Integration
Priority `claude → openai → ollama` (`tools/llm_client.py`).
- **Claude (primary):** persona-grounded dialogue, research-proposal synthesis.
- **OpenAI (fallback):** same prompts on transient Claude failure.
- **Ollama (offline/privacy):** `PRIVACY_MODE=true` forces fully-local inference.

## Knowledge Crawl Sources
ArXiv `cs.CL`, `cs.SD`, `cs.HC`, `eess.AS` + Semantic Scholar. **Daily** (Cluster A
is fast-moving). Scored by recency (≤90 days highest) × domain-keyword relevance;
deduped by SHA256 URL hash; appended to `SECOND-KNOWLEDGE-BRAIN.md`.

## Supporting Tools (`tools/`)
- `knowledge_updater.py` — research-paper crawl pipeline (REQUIRED).
- `llm_client.py` — Claude/OpenAI/Ollama unified client (REQUIRED).
- `hf_model_manager.py` — singleton lazy HuggingFace model loader.

## Active Development Tasks
- [x] Cluster A modules (stream_ingestor, speech_processor, media_synthesizer, improvement_proposer, benchmark_runner)
- [x] Orchestrator turn loop + optimization loop
- [x] Memory manager (SQLite WAL, 4 tables)
- [x] Three universal tools (knowledge_updater, llm_client, hf_model_manager)
- [x] Config, Docker, tests (47 passing), docs, fork documentation
- [ ] Wire AI layer into the live upstream frontend (Phase 7)
- [ ] Replace Whisper backend with faster-whisper per proposal #1, re-benchmark
- [ ] Collect human MOS ratings to calibrate the objective MOS proxy
