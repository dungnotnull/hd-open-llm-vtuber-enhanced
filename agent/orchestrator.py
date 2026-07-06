"""orchestrator.py — core decision loop for open-llm-vtuber-enhanced.

Wires the modules into the VTuber turn loop and the research/optimization loop.

Turn loop (real-time interaction):
  ingest -> ASR+emotion -> recall memory -> persona LLM reply ->
  TTS+Live2D expression -> persist turn -> return frontend command.

Optimization loop (the "research agent"):
  crawl papers -> benchmark current pipeline -> propose cited improvements ->
  (operator reviews / applies) -> re-benchmark -> keep only gated wins.

The orchestrator owns lazy singletons for every module/tool so importing the
package is cheap and heavy models load on first use only.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("vtuber.orchestrator")

PERSONA_SYSTEM_TEMPLATE = """You are {name}, an AI VTuber with this persona:
{persona}

Rules:
- Stay fully in character; warm, expressive, concise (1-3 sentences).
- You are speaking aloud, so avoid markdown, emojis, code, or URLs.
- Use the conversation memory to stay consistent with the viewer over time.
- Detected viewer emotion this turn: {emotion}. Respond empathetically."""


class VTuberOrchestrator:
    def __init__(self, config: Optional[Dict[str, Any]] = None, base_dir: Optional[str] = None):
        self.config = config or {}
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.brain_path = os.path.join(self.base_dir, "SECOND-KNOWLEDGE-BRAIN.md")

        persona_cfg = self.config.get("persona", {})
        self.persona_name = persona_cfg.get("name", "Aria")
        self.persona_desc = persona_cfg.get("description",
                                            "A cheerful, curious anime VTuber who loves chatting with viewers.")
        self.speaker_wav = persona_cfg.get("reference_voice") or os.getenv("REFERENCE_VOICE_WAV")

        # lazy singletons
        self._memory = None
        self._llm = None
        self._hf = None
        self._ingestor = None
        self._speech = None
        self._synth = None
        self._proposer = None
        self._bench = None
        self._knowledge = None
        self._scheduler = None

        # prometheus-style counters + gauges
        self._counters = {"turns": 0, "asr_fallbacks": 0, "tts_fallbacks": 0,
                          "llm_fallbacks": 0, "proposals": 0, "knowledge_runs": 0,
                          "benchmarks": 0}
        self._gauges: Dict[str, float] = {
            "asr_wer": 0.0, "tts_mos": 0.0,
            "turn_latency_p95_ms": 0.0, "first_token_ms": 0.0,
            "human_mos": 0.0,
        }

    # -- lazy getters ---------------------------------------------------------
    def memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            db = self.config.get("memory", {}).get("db_path",
                                                   os.path.join(self.base_dir, "data", "vtuber_memory.db"))
            self._memory = MemoryManager(db)
        return self._memory

    def llm(self):
        if self._llm is None:
            from tools.llm_client import LLMClient, LLMConfig
            lc = self.config.get("llm", {})
            cfg = LLMConfig(
                claude_model=lc.get("claude_model", "claude-opus-4-8"),
                openai_model=lc.get("openai_model", "gpt-4o"),
                ollama_model=lc.get("ollama_model", "llama3"),
                max_tokens=lc.get("max_tokens", 512),
                temperature=lc.get("temperature", 0.8))
            self._llm = LLMClient(cfg, cost_logger=self.memory().log_llm_cost)
        return self._llm

    def hf(self):
        if self._hf is None:
            from tools.hf_model_manager import get_manager
            self._hf = get_manager()
        return self._hf

    def ingestor(self):
        if self._ingestor is None:
            from agent.modules.stream_ingestor import StreamIngestor
            a = self.config.get("audio", {})
            self._ingestor = StreamIngestor(
                sample_rate=a.get("sample_rate", 16000),
                silence_threshold=a.get("silence_threshold", 0.005),
                work_dir=os.path.join(self.base_dir, "data", "captures"))
        return self._ingestor

    def speech(self):
        if self._speech is None:
            from agent.modules.speech_processor import SpeechProcessor
            self._speech = SpeechProcessor(
                hf_manager=self.hf(),
                enable_diarization=self.config.get("speech", {}).get("diarization", False))
        return self._speech

    def synth(self):
        if self._synth is None:
            from agent.modules.media_synthesizer import MediaSynthesizer
            self._synth = MediaSynthesizer(
                hf_manager=self.hf(), speaker_wav=self.speaker_wav,
                out_dir=os.path.join(self.base_dir, "data", "tts_out"),
                language=self.config.get("tts", {}).get("language", "en"))
        return self._synth

    def proposer(self):
        if self._proposer is None:
            from agent.modules.improvement_proposer import ImprovementProposer
            self._proposer = ImprovementProposer(
                self.brain_path, llm_client=self.llm(), hf_manager=self.hf())
        return self._proposer

    def bench(self):
        if self._bench is None:
            from agent.modules.benchmark_runner import BenchmarkRunner
            self._bench = BenchmarkRunner(memory=self.memory())
        return self._bench

    def knowledge(self):
        if self._knowledge is None:
            from tools.knowledge_updater import KnowledgeUpdater
            self._knowledge = KnowledgeUpdater(
                self.brain_path, memory=self.memory(),
                summarizer=self.hf().summarize)
        return self._knowledge

    # -- turn loop ------------------------------------------------------------
    def handle_turn(self, *, text: Optional[str] = None, audio_path: Optional[str] = None,
                    user_id: str = "default", session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process one user turn end-to-end. Returns the frontend command plus
        timing/metadata."""
        session_id = session_id or str(uuid.uuid4())[:8]
        t0 = time.time()
        latencies: Dict[str, float] = {}

        # 1) ingest
        if text is not None:
            ingest = self.ingestor().ingest_text(text)
        elif audio_path is not None:
            ingest = self.ingestor().ingest_file(audio_path)
        else:
            return {"error": "no input provided", "session_id": session_id}
        if not ingest.valid:
            return {"error": f"input rejected: {ingest.reason}", "session_id": session_id}

        # 2) ASR + emotion
        if ingest.modality == "text":
            speech = self.speech().process_text(ingest.text)
        else:
            speech = self.speech().process_audio(ingest.audio_path)
        latencies["asr"] = speech.asr_latency_ms
        if speech.asr_fallback:
            self._counters["asr_fallbacks"] += 1
        if not speech.confident:
            return {"error": "could not understand audio, please repeat",
                    "reason": speech.reason, "session_id": session_id}

        user_text = speech.transcript
        self.memory().save_turn(session_id, "user", user_text, user_id=user_id,
                                emotion=speech.emotion)

        # 3) recall + persona reply
        history = self.memory().recent_turns(user_id=user_id, limit=10)
        reply, llm_latency, llm_fallback = self._generate_reply(
            user_text, speech.emotion, history)
        latencies["llm"] = llm_latency
        if llm_fallback:
            self._counters["llm_fallbacks"] += 1

        # 4) synthesize speech + avatar
        synth = self.synth().synthesize(reply, emotion=speech.emotion, session_id=session_id)
        latencies["tts"] = synth.tts_latency_ms
        if synth.tts_fallback:
            self._counters["tts_fallbacks"] += 1

        latencies["turn"] = (time.time() - t0) * 1000.0
        self.memory().save_turn(session_id, "assistant", reply, user_id=user_id,
                                emotion=synth.emotion, latencies=latencies)
        self._counters["turns"] += 1
        self._gauges["turn_latency_p95_ms"] = max(self._gauges["turn_latency_p95_ms"],
                                                 latencies["turn"])
        self._gauges["first_token_ms"] = latencies.get("llm", 0.0)

        command = self.synth().build_frontend_command(synth, reply)
        return {
            "session_id": session_id,
            "user_text": user_text,
            "reply": reply,
            "emotion": speech.emotion,
            "command": command,
            "latencies_ms": {k: round(v, 1) for k, v in latencies.items()},
        }

    def _generate_reply(self, user_text: str, emotion: str,
                        history: List[Dict[str, Any]]):
        system = PERSONA_SYSTEM_TEMPLATE.format(
            name=self.persona_name, persona=self.persona_desc, emotion=emotion)
        context = "\n".join(f"{h['role']}: {h['text']}" for h in history[-8:])
        prompt = (f"Conversation so far:\n{context}\n\n"
                  f"Viewer just said: \"{user_text}\"\n\nReply in character:")
        start = time.time()
        try:
            resp = self.llm().complete(prompt, system=system, max_tokens=200, temperature=0.85)
            latency = (time.time() - start) * 1000.0
            if resp.error or not resp.text.strip():
                raise RuntimeError(resp.error or "empty reply")
            return resp.text.strip(), latency, resp.fallback_used
        except Exception as exc:  # noqa: BLE001
            logger.warning("reply generation failed, using canned line: %s", exc)
            canned = self._canned_reply(emotion)
            return canned, (time.time() - start) * 1000.0, True

    def _canned_reply(self, emotion: str) -> str:
        table = {
            "joy": "Ahh I'm so happy to hear that! Tell me more~",
            "sadness": "Aw, I'm here for you. Want to talk about it?",
            "anger": "I hear you. Let's take a breath together, okay?",
            "surprise": "Whoa, really?! That caught me off guard!",
            "neutral": "Mhm, I'm listening! What's on your mind?",
        }
        return table.get(emotion, table["neutral"])

    # -- optimization loop ----------------------------------------------------
    def update_knowledge(self) -> Dict[str, Any]:
        summary = self.knowledge().run_once()
        self._counters["knowledge_runs"] += 1
        return summary

    def propose_improvements(self) -> Dict[str, Any]:
        latest = self.memory().latest_benchmark()
        metrics = latest or {"asr_wer": 0.10, "tts_mos": 4.0, "turn_latency_p95_ms": 1500}
        proposals = self.proposer().propose(metrics)
        self._counters["proposals"] += len(proposals)
        return {"based_on": metrics, "proposals": [p.to_dict() for p in proposals]}

    def run_benchmark(self, label: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark the live pipeline over a set of text cases."""
        def turn_fn(case):
            out = self.handle_turn(text=case.get("text", ""), user_id="benchmark")
            cmd = out.get("command", {})
            return {
                "transcript": out.get("user_text", ""),
                "turn_latency_ms": out.get("latencies_ms", {}).get("turn", 0.0),
                "first_token_ms": out.get("latencies_ms", {}).get("llm", 0.0),
                "tts_fallback": cmd.get("audio") is None,
                "has_audio": cmd.get("audio") is not None,
            }
        result = self.bench().run(label, cases, turn_fn)
        self._counters["benchmarks"] += 1
        self._gauges["asr_wer"] = float(result.asr_wer or 0.0)
        self._gauges["tts_mos"] = float(result.tts_mos or 0.0)
        self._gauges["turn_latency_p95_ms"] = float(result.turn_latency_p95_ms or 0.0)
        return result.to_metrics() | {"label": label, "samples": result.samples}

    # -- scheduler / metrics --------------------------------------------------
    def start_scheduler(self, cron: str = "daily"):
        if self._scheduler is not None:
            return self._scheduler
        self._scheduler = self.knowledge().start_scheduler(cron=cron)
        return self._scheduler

    def scheduler_status(self) -> Dict[str, Any]:
        running = self._scheduler is not None and getattr(self._scheduler, "running", False)
        jobs = []
        if self._scheduler is not None:
            try:
                jobs = [str(j) for j in self._scheduler.get_jobs()]
            except Exception:  # noqa: BLE001
                jobs = []
        return {"running": bool(running), "jobs": jobs}

    def prometheus_metrics(self) -> str:
        """Render counters + gauges in Prometheus exposition text format."""
        lines = [
            "# HELP vtuber_turns_total Total VTuber turns processed end-to-end.",
            "# TYPE vtuber_turns_total counter",
            f"vtuber_turns_total {self._counters['turns']}",
            "# HELP vtuber_asr_fallbacks_total Total ASR calls that fell back to heuristic.",
            "# TYPE vtuber_asr_fallbacks_total counter",
            f"vtuber_asr_fallbacks_total {self._counters['asr_fallbacks']}",
            "# HELP vtuber_tts_fallbacks_total Total TTS calls that fell back to silence.",
            "# TYPE vtuber_tts_fallbacks_total counter",
            f"vtuber_tts_fallbacks_total {self._counters['tts_fallbacks']}",
            "# HELP vtuber_llm_fallbacks_total Total LLM calls that fell back to canned replies.",
            "# TYPE vtuber_llm_fallbacks_total counter",
            f"vtuber_llm_fallbacks_total {self._counters['llm_fallbacks']}",
            "# HELP vtuber_proposals_total Total research-driven upgrade proposals generated.",
            "# TYPE vtuber_proposals_total counter",
            f"vtuber_proposals_total {self._counters['proposals']}",
            "# HELP vtuber_knowledge_runs_total Total knowledge-base crawl runs executed.",
            "# TYPE vtuber_knowledge_runs_total counter",
            f"vtuber_knowledge_runs_total {self._counters['knowledge_runs']}",
            "# HELP vtuber_benchmarks_total Total benchmark suite runs executed.",
            "# TYPE vtuber_benchmarks_total counter",
            f"vtuber_benchmarks_total {self._counters['benchmarks']}",
            "# HELP vtuber_asr_wer Last recorded ASR word error rate.",
            "# TYPE vtuber_asr_wer gauge",
            f"vtuber_asr_wer {self._gauges['asr_wer']}",
            "# HELP vtuber_tts_mos Last recorded TTS MOS proxy.",
            "# TYPE vtuber_tts_mos gauge",
            f"vtuber_tts_mos {self._gauges['tts_mos']}",
            "# HELP vtuber_turn_latency_p95_ms Last recorded end-to-end p95 turn latency (ms).",
            "# TYPE vtuber_turn_latency_p95_ms gauge",
            f"vtuber_turn_latency_p95_ms {self._gauges['turn_latency_p95_ms']}",
            "# HELP vtuber_first_token_ms Last recorded LLM first-token latency (ms).",
            "# TYPE vtuber_first_token_ms gauge",
            f"vtuber_first_token_ms {self._gauges['first_token_ms']}",
            "# HELP vtuber_human_mos Mean human MOS rating collected for calibration.",
            "# TYPE vtuber_human_mos gauge",
            f"vtuber_human_mos {self._gauges['human_mos']}",
        ]
        return "\n".join(lines) + "\n"

    def stats(self) -> Dict[str, Any]:
        mos = self.memory().mos_summary()
        self._gauges["human_mos"] = mos.get("avg_human_mos") or 0.0
        return {"persona": self.persona_name, "counters": self._counters,
                "gauges": self._gauges, "scheduler": self.scheduler_status(),
                "memory": self.memory().get_stats()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = VTuberOrchestrator()
    out = orch.handle_turn(text="Hi! I just got back from a great trip, I'm so happy!")
    print("reply:", out.get("reply"))
    print("emotion:", out.get("emotion"), "latencies:", out.get("latencies_ms"))
