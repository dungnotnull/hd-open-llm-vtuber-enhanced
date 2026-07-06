# SECOND-KNOWLEDGE-BRAIN — open-llm-vtuber-enhanced

Self-improving domain knowledge base for the real-time AI VTuber pipeline
(streaming ASR → persona LLM → zero-shot TTS → emotion-aware Live2D). The
research agent (`tools/knowledge_updater.py`) appends new papers here daily;
`agent/modules/improvement_proposer.py` retrieves from this file to ground its
cited optimization proposals.

---

## Core Concepts & Frameworks

- **End-to-end turn latency** = ASR + LLM (first token → full reply) + TTS +
  render. The perceived responsiveness is dominated by *first-audio* latency, so
  pipelining (stream LLM tokens into incremental TTS) matters more than raw
  throughput.
- **Streaming ASR** — chunked/partial-hypothesis decoding so transcription starts
  before the user stops speaking; trade-off between chunk size, WER, and latency.
- **Zero-shot voice cloning TTS** — synthesize a target voice from a short (≈6s)
  reference clip without per-speaker fine-tuning (XTTS-v2, CosyVoice).
- **Emotion-aware avatar** — map detected dialogue emotion to Live2D expression /
  motion parameters and to TTS prosody for congruent affect.
- **Viseme / lip-sync** — align mouth-shape parameters to phoneme/viseme timing
  from the TTS engine; fall back to text-length estimation when timing is absent.
- **Persona grounding + long-term memory** — a stable system persona plus recalled
  prior turns keep the character consistent across sessions.
- **Quality gate** — accept a pipeline change only if it improves latency OR
  quality (WER/MOS) **without regressing the other**.

## Key Research Papers (seed set)

| Title | Authors | Year | Venue | Link | Key Finding | Relevance |
|-------|---------|------|-------|------|-------------|-----------|
| Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford et al. | 2022 | OpenAI | https://arxiv.org/abs/2212.04356 | Weakly-supervised multilingual ASR, robust zero-shot WER | Input ASR stage |
| XTTS: a Massively Multilingual Zero-Shot TTS | Casanova et al. | 2024 | Coqui/Interspeech | https://arxiv.org/abs/2406.04904 | 6s zero-shot voice cloning across 17 languages | Output TTS stage |
| CosyVoice: Scalable Multilingual Zero-shot TTS | Du et al. | 2024 | Alibaba | https://arxiv.org/abs/2407.05407 | Supervised semantic tokens improve naturalness/stability | TTS alternative |
| pyannote.audio 3.1 speaker diarization | Bredin | 2023 | Interspeech | https://arxiv.org/abs/2310.13025 | SOTA open diarization pipeline | Multi-speaker input |
| Moshi: a speech-text foundation model for real-time dialogue | Défossez et al. | 2024 | Kyutai | https://arxiv.org/abs/2410.00037 | Full-duplex 160ms-latency spoken dialogue | Latency / turn-taking |
| Emotion DistilRoBERTa for text classification | Hartmann | 2022 | HF | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base | Fast 7-class emotion classifier | Emotion→avatar |
| BGE: BAAI General Embedding | Xiao et al. | 2023 | BAAI | https://arxiv.org/abs/2309.07597 | Strong general-purpose retrieval embeddings | Paper retrieval |
| BART: Denoising Seq2Seq Pretraining | Lewis et al. | 2019 | ACL | https://arxiv.org/abs/1910.13461 | Strong abstractive summarization | Abstract compression |
| StreamSpeech: Simultaneous Speech-to-Speech Translation | Zhang et al. | 2024 | ACL | https://arxiv.org/abs/2406.03049 | Unified streaming S2ST with low latency | Streaming pipeline |
| VITS: Conditional VAE with Adversarial Learning for TTS | Kim et al. | 2021 | ICML | https://arxiv.org/abs/2106.06103 | End-to-end parallel TTS, high quality | TTS backbone lineage |
| FastConformer: faster ASR encoder | Rekesh et al. | 2023 | ASRU | https://arxiv.org/abs/2305.05084 | 2.4x faster Conformer, low-latency ASR | ASR speedup |
| Live2D-style talking head animation survey | Various | 2023 | — | https://arxiv.org/abs/2305.18891 | Audio-driven facial animation methods | Avatar animation |
| Whisper-Streaming / whisper_streaming | Macháček et al. | 2023 | IWSLT | https://arxiv.org/abs/2307.14743 | Real-time Whisper with local agreement policy | Streaming ASR |
| Generative Agents: Interactive Simulacra | Park et al. | 2023 | UIST | https://arxiv.org/abs/2304.03442 | Memory + reflection for believable agents | Long-term memory |
| LLaMA-Omni: Seamless Speech Interaction | Fang et al. | 2024 | — | https://arxiv.org/abs/2409.06666 | Low-latency speech-in/speech-out with LLM | Duplex dialogue |

## State-of-the-Art Models (current best)

| Task | Model | Benchmark | As of |
|------|-------|-----------|-------|
| ASR | openai/whisper-large-v3 | ~7.5% avg multilingual WER | 2024 |
| ASR (fast) | faster-whisper (CTranslate2) | ~4x speed, equal WER | 2024 |
| Zero-shot TTS | coqui/XTTS-v2 | MOS ≈ 4.1 expressive | 2024 |
| Zero-shot TTS | CosyVoice / CosyVoice2 | MOS ≈ 4.2 | 2024 |
| Diarization | pyannote/speaker-diarization-3.1 | DER ≈ 11% (DIHARD) | 2023 |
| Emotion | j-hartmann/emotion-english-distilroberta-base | ~66% 7-class acc | 2022 |
| Embeddings | BAAI/bge-large-en-v1.5 | MTEB ≈ 64 | 2023 |
| Duplex speech LLM | Moshi | 160–200ms theoretical latency | 2024 |

## LLM Prompt Patterns

1. **Persona dialogue** — system persona + last N turns + detected emotion;
   constrain to spoken-friendly, 1–3 sentences, no markdown/URLs.
2. **Proposal synthesis** — metrics block + weakest area + retrieved papers →
   JSON array of proposals, each requiring an arXiv/DOI citation.
3. **Benchmark narrative** — baseline vs candidate metric table → 3-sentence
   plain-language verdict.
4. **Relevance scoring** — given a metric deficit, rank candidate papers by how
   directly they address that deficit.

## Authoritative Data Sources

- ArXiv API — `cs.CL`, `cs.SD`, `cs.HC`, `eess.AS`
- Semantic Scholar Graph API
- Papers with Code leaderboards (ASR, TTS, speech)
- Interspeech / ICASSP proceedings
- HuggingFace Hub model cards + HF Papers daily
- Open-LLM-VTuber GitHub releases & docs

## Self-Update Protocol

```yaml
crawler: tools/knowledge_updater.py
schedule: "daily 06:00 local"          # Cluster A fast-moving domain
sources:
  arxiv_categories: [cs.CL, cs.SD, cs.HC, eess.AS]
  semantic_scholar: true
queries:
  - "streaming speech recognition low latency"
  - "zero-shot voice cloning text-to-speech"
  - "emotion-aware talking avatar animation"
  - "persona-grounded dialogue large language model"
  - "real-time conversational AI turn-taking latency"
scoring: "0.6*recency + 0.4*relevance"   # recency window 90 days
dedup: "sha256(url) checked against memory.knowledge_hashes"
top_n: 8
append_to: SECOND-KNOWLEDGE-BRAIN.md
```

## Knowledge Update Log

- **2026-06-23** — Seeded knowledge base with 15 foundational papers (ASR, TTS,
  diarization, duplex dialogue, emotion, retrieval, avatar animation), 8 SotA
  model entries, 4 LLM prompt patterns, and the daily self-update protocol.
  Crawler ready; first automated `update-knowledge` run will append below.
