# Test Scenarios — open-llm-vtuber-enhanced

End-to-end scenarios exercised by the agent. Each lists the trigger, the
expected pipeline behavior, and the observable output. Automated coverage lives
in `tests/test_agent.py`; these scenarios describe the behavior those tests and
manual runs validate.

---

## Scenario 1 — Golden path: happy text turn
**Trigger:** Viewer types `"Hi! I just got back from a great trip, I'm so happy!"`
**Pipeline:**
1. `stream_ingestor.ingest_text` → valid text.
2. `speech_processor.process_text` → emotion classified `joy`.
3. `memory_manager.recent_turns` → recalls prior context for this viewer.
4. `orchestrator._generate_reply` → persona LLM (Claude) produces an in-character,
   spoken-friendly reply.
5. `media_synthesizer.synthesize` → XTTS-v2 audio + `exp_joy` expression + `happy`
   motion + viseme lip-sync timeline.
6. Turn persisted with full latency breakdown.
**Expected output:** `command.type == "speak"`, `expression == "exp_joy"`,
non-empty `reply`, `latencies_ms.turn` recorded.

## Scenario 2 — Microphone turn with Whisper ASR
**Trigger:** `python -m agent.main listen --seconds 5` (viewer speaks).
**Pipeline:** capture → quality gate (RMS/duration) → Whisper-large-v3 transcript →
emotion → reply → TTS+avatar.
**Expected output:** JSON turn payload with transcribed `user_text` and a reply.
Falls back to `"could not understand audio, please repeat"` on an empty transcript.

## Scenario 3 — Silent / bad capture rejected before ASR
**Trigger:** A silent or 0.05s WAV is ingested.
**Pipeline:** `stream_ingestor` quality gate fails (`silent` / `too short`).
**Expected output:** `{"error": "input rejected: silent (rms=...)"}` — no Whisper
call is made (fail fast, no wasted inference).

## Scenario 4 — All LLM providers unavailable (graceful degradation)
**Trigger:** No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` and Ollama down.
**Pipeline:** `llm_client` walks the chain, every provider fails →
`orchestrator._canned_reply` returns an emotion-appropriate canned line.
**Expected output:** A valid turn still completes; `command` is produced; the
reply is the canned line for the detected emotion. The agent never crashes.

## Scenario 5 — Emotion → avatar expression mapping
**Trigger:** Replies generated for viewer messages tagged `joy`, `sadness`,
`anger`, `surprise`.
**Pipeline:** `media_synthesizer` maps each emotion to a distinct Live2D
`expression`/`motion`/`live2d_params` set.
**Expected output:** `joy → exp_joy/happy`, `sadness → exp_sad/sad`,
`anger → exp_angry/angry`, unknown emotion → `exp_neutral`.

## Scenario 6 — Research crawler appends new papers, dedups on re-run
**Trigger:** `python -m agent.main update-knowledge` run twice.
**Pipeline:** ArXiv (cs.CL/cs.SD/cs.HC/eess.AS) + Semantic Scholar fetch → score
by recency×relevance → SHA256 dedup against `knowledge_hashes` → append top-N to
`SECOND-KNOWLEDGE-BRAIN.md`.
**Expected output:** First run appends N new rows; second run reports `new: 0`
(all URLs already known) — proving deduplication works.

## Scenario 7 — Cited optimization proposals from the knowledge base
**Trigger:** `python -m agent.main propose` after a benchmark recorded weak metrics.
**Pipeline:** `improvement_proposer` picks the weakest metric → retrieves relevant
papers (BGE embed + rerank) → LLM synthesizes 3-5 JSON proposals, each requiring
an arXiv/DOI citation → falls back to 3 durable real-arXiv proposals if LLM/KB
unavailable.
**Expected output:** ≥3 proposals, each with `target_metric`, `expected_impact`,
and a `citation` starting with `http`.

## Scenario 8 — Before/after benchmark gate
**Trigger:** Run `benchmark` to set a baseline, apply a proposal, run again.
**Pipeline:** `benchmark_runner` measures WER / MOS-proxy / p50 / p95 / first-token
over the case set, persists each run, then `evaluate_gates` compares candidate vs
baseline.
**Expected output:** PASS only when the candidate improves latency or MOS or WER
**without regressing** the others; a Markdown report with explicit ✅/❌ verdicts.

---

### Coverage map (automated)
| Module | Tests |
|--------|-------|
| stream_ingestor | 6 |
| speech_processor | 3 |
| media_synthesizer | 4 |
| improvement_proposer | 4 |
| benchmark_runner | 6 |
| memory_manager | 5 |
| llm_client | 3 |
| hf_model_manager | 4 |
| knowledge_updater | 3 |
| integration (orchestrator E2E) | 6 |
| CLI smoke | 3 |
| **Total** | **47** |
