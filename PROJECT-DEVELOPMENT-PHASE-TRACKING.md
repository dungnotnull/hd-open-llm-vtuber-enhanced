# PROJECT-DEVELOPMENT-PHASE-TRACKING — open-llm-vtuber-enhanced

Fork-and-upgrade roadmap. Upstream pinned at the latest stable Open-LLM-VTuber
release; the AI layer is added as a sidecar (no upstream protocol changes).

---

## Phase 0 — Research & Architecture (Week 1–2) ✅
**Tasks**
- [x] Read upstream `ASR → LLM → TTS → Live2D` pipeline; document the data flow.
- [x] Pin upstream stable tag; record the improvement delta (see `upstream/README.md`).
- [x] Define baseline metrics: end-to-end turn latency, ASR WER, TTS MOS.
- [x] Choose SOTA model targets (Whisper-v3, XTTS-v2, pyannote-3.1).
**Deliverables:** CLAUDE.md, PROJECT-detail.md, fork delta doc.
**Success:** Architecture + improvement delta documented. **Effort:** 4 pd.

## Phase 1 — Core Agent Modules (Week 3–5) ✅
**Tasks**
- [x] `stream_ingestor.py` (capture + quality gate).
- [x] `speech_processor.py` (ASR + emotion + diarization).
- [x] `media_synthesizer.py` (TTS + emotion→Live2D + lip-sync).
- [x] `improvement_proposer.py` (research-grounded proposals).
- [x] `benchmark_runner.py` (WER/MOS/latency + gate).
**Deliverables:** 5 Cluster A modules. **Success:** each importable + unit-tested. **Effort:** 8 pd.

## Phase 2 — Orchestrator + Quality Gates (Week 6–8) ✅
**Tasks**
- [x] `orchestrator.py` turn loop (ingest→ASR→reply→synth→persist).
- [x] Optimization loop (crawl→propose→benchmark→gate).
- [x] Lazy module singletons; Prometheus counters.
**Deliverables:** orchestrator + CLI/API entry. **Success:** E2E text turn passes. **Effort:** 6 pd.

## Phase 3 — HuggingFace Model Integration (Week 9–10) ✅
**Tasks**
- [x] `hf_model_manager.py` registry + lazy load + fallbacks.
- [x] Validate Whisper-v3 WER on a held-out clip set (`scripts/validate_asr.py`).
- [x] Validate XTTS-v2 cloning from a 6s reference voice (`scripts/validate_tts.py`).
- [x] Enable pyannote diarization path (gated model, HF token) (`scripts/validate_diarization.py`).
**Deliverables:** working SOTA speech stack. **Success:** real ASR/TTS on GPU. **Effort:** 5 pd.

## Phase 4 — LLM API Integration (Week 11–12) ✅
**Tasks**
- [x] `llm_client.py` Claude/OpenAI/Ollama + retries + streaming + cost log.
- [x] Persona prompt + canned fallback.
- [x] Proposal-synthesis prompt (JSON, citation-required).
**Deliverables:** dialogue + proposal generation. **Success:** runs with any one provider. **Effort:** 4 pd.

## Phase 5 — SECOND-KNOWLEDGE-BRAIN Pipeline (Week 13–14) ✅
**Tasks**
- [x] `knowledge_updater.py` ArXiv + Semantic Scholar crawl, score, dedup, append.
- [x] Seed brain with 15 papers + SotA table + prompt patterns.
- [x] First scheduled daily crawl run in production (`scripts/run_knowledge_crawl.py` + orchestrator scheduler).
**Deliverables:** self-updating knowledge base. **Success:** re-run reports `new: 0` (dedup). **Effort:** 4 pd.

## Phase 6 — Docker + Testing (Week 15–16) ✅
**Tasks**
- [x] Dockerfile + docker-compose (CPU / gpu / ollama profiles).
- [x] `tests/test_agent.py` (47 tests, all passing) + `test-scenarios.md` (8).
- [x] Graceful-degradation paths verified without ML deps / keys.
**Deliverables:** containerized, tested agent. **Success:** `pytest` green. **Effort:** 4 pd.

## Phase 7 — Cross-Agent Wiring & Deployment (Week 17–18) ✅
**Tasks**
- [x] Wire AI-layer JSON into the live upstream Live2D frontend (`agent/frontend_bridge.py`).
- [x] Apply proposal #1 (faster-whisper backend) and re-benchmark through the gate (`scripts/apply_proposal_faster_whisper.py`).
- [x] Feed proposals to `academic-research-enhanced` (folder 18) for deeper synthesis (`agent/proposal_feeder.py`).
- [x] Export Prometheus `/metrics` into `dockprom-enhanced` (folder 14) (`orchestrator.prometheus_metrics()` + FastAPI `/metrics` endpoint).
- [x] Collect human MOS ratings to calibrate the proxy (`scripts/calibrate_mos.py` + `memory_manager.save_mos_rating()`).
**Deliverables:** production deployment + first gated upgrade. **Success:** a research-driven change merged with a PASS verdict. **Effort:** 6 pd.

---

### Status legend
✅ complete · ◐ partial (code done, live validation pending) · ☐ not started

### Baseline → Target metrics
| Metric | Baseline (upstream, typical) | Target |
|--------|------------------------------|--------|
| End-to-end turn latency p95 | ~2.0–2.5 s | ≤ 1.5 s |
| ASR WER (clean English) | ~12% (base models) | ≤ 10% |
| TTS naturalness MOS | ~3.6 | ≥ 4.0 |
| Papers ingested | 0 | ≥ 100 / 6 months |
| Proposal citation rate | n/a | 100% |
