# open-llm-vtuber-enhanced

[![Tests](https://img.shields.io/badge/tests-69%20passing-success)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-production--ready-green)](https://github.com)

**Real-time interactive AI VTuber & voice companion with autonomous research-driven optimization.**

A fork and enhancement of [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) that adds a SOTA speech stack, emotion-aware avatar, persona-grounded dialogue with long-term memory, and an autonomous research agent that continuously proposes cited pipeline optimizations.

## Overview

This project implements a complete AI VTuber pipeline with:

- **SOTA Speech Stack**: Whisper-large-v3 ASR, XTTS-v2 zero-shot voice cloning, pyannote diarization
- **Emotion-Aware Avatar**: Real-time emotion detection mapped to Live2D expressions and motion
- **Persona Dialogue**: Grounded long-term memory with consistent character personality
- **Autonomous Optimization**: Research agent that crawls latest papers and proposes cited improvements
- **Quality Gates**: Benchmark-driven validation that accepts only latency/quality improvements

## Architecture

```
User Input → Stream Ingestor → Speech Processor (ASR + Emotion) 
    ↓
Memory Recall → LLM Client (Claude/OpenAI/Ollama) → Persona Reply
    ↓
Media Synthesizer (TTS + Live2D Expression + Lip-sync) → Frontend
    ↓
Persistent Memory + Prometheus Metrics
    ↓
[Optimization Loop] → Paper Crawler → Proposal Generator → Benchmark → Quality Gate
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- GPU with CUDA (optional, for acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/open-llm-vtuber-enhanced.git
cd open-llm-vtuber-enhanced

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

### Configuration

Edit `config/.env` with your API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...  # For gated models like pyannote
REFERENCE_VOICE_WAV=/path/to/6s_reference.wav
```

### Running

#### Interactive Chat (CLI)

```bash
python -m agent.main chat
```

#### Voice Interaction

```bash
python -m agent.main listen --seconds 5
```

#### FastAPI Server

```bash
python -m agent.main serve --host 0.0.0.0 --port 8001
```

#### Frontend Bridge (WebSocket)

```bash
python -m agent.main bridge --port 8002
```

### Docker Deployment

```bash
# CPU-only deployment
docker-compose -f docker/docker-compose.yml up -d

# GPU deployment
docker-compose -f docker/docker-compose.yml --profile gpu up -d

# With Ollama (offline/privacy mode)
docker-compose -f docker/docker-compose.yml --profile ollama up -d
```

## API Endpoints

### REST API (Port 8001)

- `POST /api/v1/turn` — Process a user turn (text or audio)
- `POST /api/v1/knowledge/update` — Trigger research paper crawl
- `GET /api/v1/proposals` — Get optimization proposals
- `POST /api/v1/benchmark` — Run benchmark suite
- `GET /api/v1/stats` — Get agent statistics
- `GET /metrics` — Prometheus metrics endpoint

### WebSocket Bridge (Port 8002)

Connect to `ws://localhost:8002/vtuber` and send:

```json
{
  "type": "user_input",
  "text": "Hello VTuber!",
  "user_id": "viewer123"
}
```

## Modules

### Core Agent Modules (`agent/modules/`)

- **`stream_ingestor.py`** — Audio/text capture with quality gates
- **`speech_processor.py`** — Whisper ASR + emotion classification + diarization
- **`media_synthesizer.py`** — XTTS-v2 TTS + Live2D expression + lip-sync
- **`improvement_proposer.py`** — Research-grounded optimization proposals
- **`benchmark_runner.py`** — WER/MOS/latency measurement + quality gates

### Tools (`tools/`)

- **`hf_model_manager.py`** — Unified HuggingFace model registry with backend switching
- **`llm_client.py`** — Claude/OpenAI/Ollama client with automatic fallback
- **`knowledge_updater.py`** — ArXiv/Semantic Scholar paper crawler

### Orchestrator (`agent/`)

- **`orchestrator.py`** — Turn loop + optimization loop coordination
- **`memory_manager.py`** — SQLite persistent memory (conversations, benchmarks, costs)
- **`main.py`** — CLI + FastAPI entry point
- **`frontend_bridge.py`** — WebSocket bridge to Live2D frontend

## Validation & Benchmarking

### Run Validation Scripts

```bash
# Validate ASR WER
python -m scripts.validate_asr --manifest data/clips.jsonl

# Validate TTS cloning
python -m scripts.validate_tts --reference voice.wav

# Validate diarization
python -m scripts.validate_diarization --audio room.wav

# Apply proposal #1 (faster-whisper) and re-benchmark
python -m scripts.apply_proposal_faster_whisper

# Calibrate MOS proxy against human ratings
python -m scripts.calibrate_mos
```

## Quality Metrics

The pipeline targets these metrics:

| Metric | Baseline | Target |
|--------|----------|--------|
| End-to-end turn latency p95 | ~2.0–2.5 s | ≤ 1.5 s |
| ASR WER (clean English) | ~12% | ≤ 10% |
| TTS naturalness MOS | ~3.6 | ≥ 4.0 |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_agent.py -v      # Core functionality (47 tests)
pytest tests/test_phase_3_5_7.py -v # Phase deliverables (22 tests)
```

## Development

### Project Structure

```
open-llm-vtuber-enhanced/
├── agent/                    # Core agent modules
│   ├── modules/             # Cluster A modules
│   ├── memory/              # Persistent memory
│   └── *.py                # Orchestrator & entry points
├── tools/                   # Shared tools
├── scripts/                 # Validation & operational scripts
├── tests/                   # Comprehensive test suite
├── config/                  # Configuration files
├── docker/                  # Docker deployment
└── SECOND-KNOWLEDGE-BRAIN.md # Self-updating knowledge base
```

### Adding Features

1. Implement new modules in `agent/modules/`
2. Add tests in `tests/`
3. Update `config/agent_config.yaml`
4. Run validation: `pytest tests/ -v`

## Research Agent

The autonomous research agent:

1. **Crawls** ArXiv (cs.CL, cs.SD, cs.HC, eess.AS) + Semantic Scholar daily
2. **Scores** papers by recency × relevance
3. **Deduplicates** via SHA256 URL hashes
4. **Proposes** cited optimizations grounded in retrieved papers
5. **Validates** proposals through benchmark quality gates

View the knowledge base: `SECOND-KNOWLEDGE-BRAIN.md`

## Live2D Integration

The agent produces Live2D-compatible commands:

```json
{
  "type": "avatar-speak",
  "audio": "/path/to/audio.wav",
  "text": "Hello there!",
  "expression": "exp_joy",
  "motion": "happy",
  "live2d_params": {"ParamBrowLY": 0.6, "ParamMouthForm": 1.0},
  "lip_sync": [{"start": 0.0, "end": 0.15, "mouth_open": 0.8}],
  "emotion": "joy"
}
```

## Prometheus Metrics

Available at `/metrics`:

```
vtuber_turns_total
vtuber_asr_fallbacks_total
vtuber_tts_fallbacks_total
vtuber_turn_latency_p95_ms
vtuber_asr_wer
vtuber_tts_mos
vtuber_human_mos
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `HF_TOKEN` | HuggingFace token (gated models) | — |
| `REFERENCE_VOICE_WAV` | Path to 6s reference voice | — |
| `PRIVACY_MODE` | Force Ollama (offline) | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `VTUBER_PORT` | API server port | `8001` |

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) — Upstream VTuber framework
- [Whisper](https://github.com/openai/whisper) — OpenAI speech recognition
- [XTTS-v2](https://github.com/coqui-xtts/XTTS-v2) — Coqui TTS zero-shot cloning
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Speaker diarization

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/open-llm-vtuber-enhanced/issues)
- **Documentation**: See `CLAUDE.md` for development guidelines
- **Phase Tracking**: See `PROJECT-DEVELOPMENT-PHASE-TRACKING.md`

---

**Status**: ✅ Production Ready | **Phase**: All 8 phases complete | **Tests**: 69/69 passing
