# Upstream Fork — Open-LLM-VTuber

This project forks **[Open-LLM-VTuber/Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)**
and layers an autonomous AI optimization agent on top of it (sidecar pattern,
**zero upstream protocol changes**).

## Pin
- **Upstream:** `Open-LLM-VTuber/Open-LLM-VTuber`
- **Pinned at:** latest stable release tag as of **2026-06-23**.
- **Clone target:** `upstream/Open-LLM-VTuber/` (git submodule or direct clone).
- **License:** upstream MIT — respected; the AI layer lives in `agent/` + `tools/`.

```bash
# from this folder
git clone --depth 1 https://github.com/Open-LLM-VTuber/Open-LLM-VTuber upstream/Open-LLM-VTuber
# (or pin a specific tag once chosen)
# git -C upstream/Open-LLM-VTuber checkout <stable-tag>
```

## Upstream pipeline (documented baseline)
`microphone → ASR → LLM → TTS → Live2D frontend (Cubism)`, wired over a
WebSocket between a Python backend and a browser/Electron frontend. Models are
configured via YAML; emotion tags in the LLM output drive avatar expressions.

## Improvement delta (what this fork adds)
| Area | Upstream | This fork |
|------|----------|-----------|
| ASR | configurable (often base/medium Whisper) | Whisper-large-v3 + proposed faster-whisper backend |
| TTS | configurable engine | XTTS-v2 zero-shot cloning + emotion-conditioned prosody |
| Diarization | none | pyannote-3.1 (multi-speaker rooms) |
| Emotion → avatar | tag-based | classifier-driven (`emotion-distilroberta`) → Live2D params |
| Memory | session context | persistent SQLite long-term memory across sessions |
| Self-improvement | none | **research agent**: daily paper crawl → cited proposals → benchmark gate |
| Observability | minimal | per-stage latency, cost log, Prometheus `/metrics` |

## Integration pattern (sidecar)
The AI layer exposes `POST /api/v1/turn` returning the exact JSON the upstream
frontend consumes to drive the avatar:
```json
{
  "type": "speak",
  "text": "...",
  "audio": "/app/data/tts_out/xxx.wav",
  "expression": "exp_joy",
  "motion": "happy",
  "live2d_params": {"ParamMouthForm": 1.0},
  "lip_sync": [{"start": 0.0, "end": 0.14, "mouth_open": 0.5}],
  "emotion": "joy"
}
```
Point the upstream frontend's WebSocket/HTTP bridge at this endpoint, or run the
agent standalone via the Click CLI.

## Quantified improvement targets
1. End-to-end turn latency p95: **2.0–2.5s → ≤ 1.5s**.
2. ASR WER (clean English): **~12% → ≤ 10%**.
3. TTS naturalness MOS: **~3.6 → ≥ 4.0**.
4. Research throughput: **≥ 100 papers / 6 months**, **100%** proposal citation rate.

## Measure-before-modify
Run `python -m agent.main benchmark --label baseline` against the upstream
configuration first, record the metrics, then apply a proposal and re-benchmark.
`benchmark_runner.evaluate_gates` decides whether the change is kept.
