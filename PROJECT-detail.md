# PROJECT-detail.md — open-llm-vtuber-enhanced

## Executive Summary
A fork-and-upgrade of **Open-LLM-VTuber** that turns a static voice-companion
pipeline into a *self-optimizing* one. On top of the upstream
`ASR → LLM → TTS → Live2D` loop we add a SOTA speech stack, emotion-aware avatar
control, persona-grounded dialogue with persistent memory, and — the core
differentiator from `idea.txt` — an autonomous **research agent** that ingests
the latest speech/dialogue/HCI papers and proposes *cited* optimizations, gated
by before/after benchmarks.

## Problem Statement
Real-time conversational avatars live or die by latency and naturalness, yet the
field moves monthly (faster ASR backends, better zero-shot TTS, duplex dialogue
models). A fixed pipeline silently falls behind. We need an agent that (a) runs
the avatar well today and (b) keeps proposing evidence-backed upgrades as the
research frontier moves — without regressing what already works.

## Target Users & Use Cases
- **VTuber streamers / hobbyists** — "I speak into my mic → the avatar replies in
  my chosen voice and expression with sub-1.5s latency."
- **Voice-companion builders** — "I run the agent as a sidecar and consume its
  `/api/v1/turn` JSON to drive any Live2D frontend."
- **Researchers / maintainers** — "Each week the agent shows me 3–5 cited upgrade
  proposals and a benchmark verdict on whether to merge them."

## Agent Architecture
```
mic / file / text
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ VTuberOrchestrator (agent/orchestrator.py)                   │
│                                                              │
│  stream_ingestor → speech_processor → [memory recall]        │
│        │                 │                   │               │
│   quality gate      Whisper ASR         persona LLM reply     │
│                     + emotion           (llm_client)          │
│                          │                   │               │
│                          └──────► media_synthesizer ◄────────┘
│                                   XTTS-v2 + emotion→Live2D     │
│                                        │                      │
│                                   frontend command            │
│  ── optimization loop ───────────────────────────────────    │
│  knowledge_updater → improvement_proposer → benchmark_runner  │
└──────────────────────────────────────────────────────────────┘
   │              │                 │                │
 Whisper/XTTS  Claude/GPT/Ollama  ArxXiv/S2 API   SQLite memory
 (hf_model_mgr) (llm_client)      (knowledge)     (memory_mgr)
```

## Full Module Catalog
| Module | Responsibility | Inputs | Outputs | Tools called | Quality gate |
|--------|----------------|--------|---------|--------------|--------------|
| `stream_ingestor` | Capture + validate input | mic/file/text | `IngestResult` | sounddevice/soundfile | RMS>thr, 0.2–30s |
| `speech_processor` | ASR + emotion (+diarize) | audio/text | `SpeechResult` | hf_model_manager | non-empty transcript |
| `media_synthesizer` | TTS + avatar control | reply, emotion | `SynthesisResult` + frontend cmd | hf_model_manager | audio file produced |
| `improvement_proposer` | Cited optimization proposals | metrics, brain | `Proposal[]` | llm_client, hf, knowledge | each has citation |
| `benchmark_runner` | Measure + gate | cases, turn_fn | `BenchmarkResult` + verdict | memory | improve, no regress |

## HuggingFace Model Selection
| Model | Task | Why chosen vs alternatives |
|-------|------|----------------------------|
| whisper-large-v3 | ASR | Best open multilingual WER + word timestamps (vs wav2vec2 CTC); faster-whisper is the proposed drop-in speedup. |
| XTTS-v2 | TTS | True 6s zero-shot cloning (vs Bark's weaker control, VITS' per-speaker training). |
| pyannote-3.1 | Diarization | SOTA open DER (vs NeMo, simpler clustering). |
| emotion-distilroberta | Emotion | 7-class, CPU-fast (vs heavier GoEmotions models). |
| bge-large + bge-reranker | Retrieval | Top MTEB open pair for grounding proposals. |
| bart-large-cnn | Summarization | Reliable abstractive summary for context fit. |

## LLM API Integration Spec
- Provider chain `claude → openai → ollama`, exponential backoff 1s/2s/4s.
- **Dialogue:** `max_tokens≈200`, `temperature≈0.85`, persona system prompt.
- **Proposals:** `max_tokens≈1400`, `temperature≈0.4`, JSON-only, citation-required.
- Token budget per turn ≈ 300–700 tokens; cost logged to `llm_cost_log`.
- Fallback: canned emotion-appropriate reply if all providers fail.

## E2E Execution Flow
1. Ingest → quality gate (reject silent/short/long; fail fast).
2. ASR + emotion (text path skips ASR). Low-confidence transcript → ask to repeat.
3. Persist user turn; recall last 10 turns for context.
4. Persona LLM reply (stream-capable) → on failure, canned line.
5. TTS + emotion→Live2D expression/motion + viseme timeline.
6. Persist assistant turn with `{asr, llm, tts, turn}` latencies.
7. Return frontend command JSON.
8. (async) crawler appends papers; proposer drafts cited upgrades; benchmark gates them.

## SECOND-KNOWLEDGE-BRAIN Integration
Sources: ArXiv (cs.CL/cs.SD/cs.HC/eess.AS) + Semantic Scholar. Scored
`0.6*recency + 0.4*relevance`; deduped by SHA256(url) against
`memory.knowledge_hashes`; top-N appended daily. The proposer parses the markdown
tables back out for retrieval.

## Component Specs
- **knowledge_updater.py** — stdlib HTTP, ArXiv Atom + S2 JSON parsing, scoring,
  dedup, append, APScheduler daily 06:00. Failure → logs + returns empty summary.
- **llm_client.py** — provider chain, retries, streaming, cost accounting hook,
  `PRIVACY_MODE`.
- **hf_model_manager.py** — singleton, lazy load, CUDA/MPS auto-detect, 600s idle
  unload, deterministic fallbacks per task.
- **docker-compose.yml** — `vtuber-agent` (CPU), `gpu` profile (NVIDIA passthrough),
  `ollama` profile; volumes for data/models; mounts the brain file.

## Quality Gates (≥5)
1. Input quality gate (RMS / duration) before any ASR call.
2. ASR confidence gate (non-empty transcript) before replying.
3. Proposal citation gate (every proposal carries an arXiv/DOI link).
4. Benchmark improvement gate (latency or quality up, neither regressed).
5. Absolute targets: WER ≤ 0.10, MOS ≥ 4.0, turn p95 ≤ 1500ms.
6. Knowledge dedup gate (no duplicate paper appended).

## Test Scenarios (≥5)
See `tests/test-scenarios.md` — 8 scenarios (happy turn, mic ASR, bad-capture
rejection, all-LLM-down fallback, emotion→avatar mapping, crawler dedup, cited
proposals, before/after gate). 47 automated tests in `tests/test_agent.py`.

## Key Design Decisions
1. **Sidecar over upstream rewrite** — the AI layer consumes/produces JSON the
   Open-LLM-VTuber frontend already speaks; zero protocol changes to the fork.
2. **Graceful degradation everywhere** — no API key / no GPU / no model still
   yields a valid turn via deterministic fallbacks.
3. **Citations are mandatory** — proposals without a paper link are dropped; this
   is what makes the research agent trustworthy.
4. **Gate, don't trust** — every proposed change must beat the recorded baseline
   under the benchmark before it's kept.
5. **Objective MOS proxy** — until human ratings are collected, MOS is proxied
   from synthesis success/audio presence so the gate is runnable in CI.
