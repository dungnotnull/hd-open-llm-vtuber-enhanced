"""Automated tests for open-llm-vtuber-enhanced.

These run without any ML deps or API keys: heavy models degrade to deterministic
fallbacks, the LLM client falls back through the chain, and audio probing uses
the stdlib. Run: pytest tests/test_agent.py
"""

import os
import sys
import wave
import struct
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_wav(path, seconds=1.0, sample_rate=16000, amplitude=8000, silent=False):
    n = int(seconds * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n):
            val = 0 if silent else int(amplitude * ((i % 100) / 100.0 - 0.5))
            wf.writeframes(struct.pack("<h", val))


# --------------------------------------------------------------------------- #
# StreamIngestor
# --------------------------------------------------------------------------- #
class TestStreamIngestor:
    def _ing(self):
        from agent.modules.stream_ingestor import StreamIngestor
        return StreamIngestor(work_dir=tempfile.mkdtemp())

    def test_text_ok(self):
        r = self._ing().ingest_text("hello there")
        assert r.valid and r.modality == "text" and r.text == "hello there"

    def test_empty_text_rejected(self):
        r = self._ing().ingest_text("   ")
        assert not r.valid

    def test_file_not_found(self):
        r = self._ing().ingest_file("/nope/missing.wav")
        assert not r.valid and "not found" in r.reason

    def test_valid_wav(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "ok.wav")
        _make_wav(p, seconds=1.0)
        r = self._ing().ingest_file(p)
        assert r.valid and r.sample_rate == 16000

    def test_silent_wav_rejected(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "silent.wav")
        _make_wav(p, seconds=1.0, silent=True)
        r = self._ing().ingest_file(p)
        assert not r.valid and "silent" in r.reason

    def test_too_short_rejected(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "short.wav")
        _make_wav(p, seconds=0.05)
        r = self._ing().ingest_file(p)
        assert not r.valid


# --------------------------------------------------------------------------- #
# SpeechProcessor
# --------------------------------------------------------------------------- #
class TestSpeechProcessor:
    def _sp(self):
        from agent.modules.speech_processor import SpeechProcessor
        from tools.hf_model_manager import get_manager
        return SpeechProcessor(hf_manager=get_manager())

    def test_process_text(self):
        r = self._sp().process_text("I'm so happy today!!")
        assert r.transcript and r.confident
        assert r.emotion  # some label

    def test_emotion_keyword_fallback(self):
        from agent.modules.speech_processor import SpeechProcessor
        sp = SpeechProcessor(hf_manager=None)
        r = sp.process_text("hello")
        assert r.emotion == "neutral"

    def test_audio_no_hf_returns_fallback(self):
        from agent.modules.speech_processor import SpeechProcessor
        sp = SpeechProcessor(hf_manager=None)
        r = sp.process_audio("anything.wav")
        assert not r.confident and r.asr_fallback


# --------------------------------------------------------------------------- #
# MediaSynthesizer
# --------------------------------------------------------------------------- #
class TestMediaSynthesizer:
    def _ms(self):
        from agent.modules.media_synthesizer import MediaSynthesizer
        return MediaSynthesizer(hf_manager=None, out_dir=tempfile.mkdtemp())

    def test_emotion_maps_to_expression(self):
        r = self._ms().synthesize("yay!", emotion="joy")
        assert r.expression == "exp_joy" and r.motion == "happy"

    def test_unknown_emotion_defaults_neutral(self):
        r = self._ms().synthesize("hi", emotion="confusion")
        assert r.expression == "exp_neutral"

    def test_visemes_generated(self):
        r = self._ms().synthesize("hello world", emotion="neutral")
        assert len(r.visemes) == 2

    def test_frontend_command_shape(self):
        ms = self._ms()
        r = ms.synthesize("hi there", emotion="joy")
        cmd = ms.build_frontend_command(r, "hi there")
        assert cmd["type"] == "speak" and "lip_sync" in cmd and cmd["expression"] == "exp_joy"


# --------------------------------------------------------------------------- #
# ImprovementProposer
# --------------------------------------------------------------------------- #
class TestImprovementProposer:
    def _pp(self, brain=None):
        from agent.modules.improvement_proposer import ImprovementProposer
        return ImprovementProposer(brain or "/nonexistent.md")

    def test_fallback_proposals_without_llm(self):
        props = self._pp().propose({"asr_wer": 0.12, "tts_mos": 3.5, "turn_latency_p95_ms": 1900})
        assert len(props) >= 3
        assert all(p.citation.startswith("http") for p in props)

    def test_weakest_area_picks_latency(self):
        from agent.modules.improvement_proposer import ImprovementProposer
        area = ImprovementProposer._weakest_area({"asr_wer": 0.02, "tts_mos": 4.5,
                                                  "turn_latency_p95_ms": 5000})
        assert "latency" in area

    def test_parse_json_requires_citation(self):
        pp = self._pp()
        good = '[{"title":"x","change":"y","citation":"https://arxiv.org/abs/1"}]'
        assert len(pp._parse_json(good)) == 1
        bad = '[{"title":"x","change":"y"}]'
        assert len(pp._parse_json(bad)) == 0

    def test_load_papers_from_brain(self, tmp_path):
        brain = tmp_path / "brain.md"
        brain.write_text(
            "| Title | Authors | Date | Source | Score | Key Finding | Link |\n"
            "|---|---|---|---|---|---|---|\n"
            "| Streaming ASR | A | 2025-01-01 | arxiv | 0.9 | low latency | https://arxiv.org/abs/1 |\n",
            encoding="utf-8")
        pp = self._pp(str(brain))
        papers = pp._load_papers()
        assert papers and papers[0]["url"].startswith("http")


# --------------------------------------------------------------------------- #
# BenchmarkRunner
# --------------------------------------------------------------------------- #
class TestBenchmarkRunner:
    def _br(self):
        from agent.modules.benchmark_runner import BenchmarkRunner
        return BenchmarkRunner()

    def test_wer_exact(self):
        assert self._br().word_error_rate("a b c", "a b c") == 0.0

    def test_wer_one_sub(self):
        assert abs(self._br().word_error_rate("a b c", "a b d") - 1 / 3) < 1e-9

    def test_percentile(self):
        assert self._br().percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_gate_pass_on_improvement(self):
        from agent.modules.benchmark_runner import BenchmarkResult
        br = self._br()
        base = BenchmarkResult("b", asr_wer=0.12, tts_mos=3.6, turn_latency_p95_ms=1800)
        cand = BenchmarkResult("c", asr_wer=0.11, tts_mos=4.1, turn_latency_p95_ms=1200)
        g = br.evaluate_gates(base, cand)
        assert g["passed"] and g["verdicts"]["shows_improvement"]

    def test_gate_fail_on_regression(self):
        from agent.modules.benchmark_runner import BenchmarkResult
        br = self._br()
        base = BenchmarkResult("b", asr_wer=0.10, tts_mos=4.2, turn_latency_p95_ms=1200)
        cand = BenchmarkResult("c", asr_wer=0.18, tts_mos=3.5, turn_latency_p95_ms=1300)
        g = br.evaluate_gates(base, cand)
        assert not g["passed"]

    def test_markdown_report(self):
        from agent.modules.benchmark_runner import BenchmarkResult
        br = self._br()
        base = BenchmarkResult("b", asr_wer=0.12, tts_mos=3.6, turn_latency_p95_ms=1800)
        cand = BenchmarkResult("c", asr_wer=0.11, tts_mos=4.1, turn_latency_p95_ms=1200)
        md = br.to_markdown(base, cand, br.evaluate_gates(base, cand))
        assert "Benchmark Report" in md and "PASS" in md


# --------------------------------------------------------------------------- #
# MemoryManager
# --------------------------------------------------------------------------- #
class TestMemoryManager:
    def _mm(self):
        from agent.memory.memory_manager import MemoryManager
        return MemoryManager(os.path.join(tempfile.mkdtemp(), "m.db"))

    def test_save_and_recall(self):
        mm = self._mm()
        mm.save_turn("s", "user", "hello", emotion="joy")
        mm.save_turn("s", "assistant", "hi!", emotion="joy")
        turns = mm.recent_turns()
        assert len(turns) == 2 and turns[0]["role"] == "user"
        mm.close()

    def test_search_memory(self):
        mm = self._mm()
        mm.save_turn("s", "user", "I love astronomy")
        assert mm.search_memory("default", "astronomy")
        mm.close()

    def test_benchmark_roundtrip(self):
        mm = self._mm()
        mm.save_benchmark("run1", {"asr_wer": 0.1, "tts_mos": 4.0,
                                   "turn_latency_p95_ms": 1400})
        assert mm.latest_benchmark()["label"] == "run1"
        mm.close()

    def test_cost_logging(self):
        mm = self._mm()
        mm.log_llm_cost({"provider": "claude", "model": "claude-opus-4-8",
                         "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01})
        s = mm.cost_summary()
        assert s["total_usd"] >= 0.01
        mm.close()

    def test_paper_dedup(self):
        mm = self._mm()
        assert not mm.is_known_paper("h1")
        mm.mark_paper_known("h1", "t", "u")
        assert mm.is_known_paper("h1")
        mm.close()


# --------------------------------------------------------------------------- #
# LLMClient
# --------------------------------------------------------------------------- #
class TestLLMClient:
    def test_cost_estimate(self):
        from tools.llm_client import LLMClient
        c = LLMClient()
        assert c._estimate_cost("claude-opus-4-8", 1000, 1000) == pytest.approx(0.09)

    def test_provider_chain_privacy(self):
        from tools.llm_client import LLMClient, LLMConfig
        c = LLMClient(LLMConfig(privacy_mode=True))
        assert c._provider_chain() == ["ollama"]

    def test_unknown_model_zero_cost(self):
        from tools.llm_client import LLMClient
        assert LLMClient()._estimate_cost("mystery", 100, 100) == 0.0


# --------------------------------------------------------------------------- #
# HFModelManager
# --------------------------------------------------------------------------- #
class TestHFModelManager:
    def test_singleton(self):
        from tools.hf_model_manager import get_manager
        assert get_manager() is get_manager()

    def test_emotion_fallback(self):
        from tools.hf_model_manager import get_manager
        scores = get_manager()._heuristic_emotion("I am so happy and love this!")
        assert "joy" in scores

    def test_viseme_estimate(self):
        from tools.hf_model_manager import HFModelManager
        v = HFModelManager._estimate_visemes("hello world")
        assert len(v) == 2 and v[0]["end"] > v[0]["start"]

    def test_hash_embed_deterministic(self):
        from tools.hf_model_manager import HFModelManager
        a = HFModelManager._hash_embed("speech recognition")
        b = HFModelManager._hash_embed("speech recognition")
        assert a == b and len(a) == 384


# --------------------------------------------------------------------------- #
# KnowledgeUpdater
# --------------------------------------------------------------------------- #
class TestKnowledgeUpdater:
    def _ku(self, tmp_path):
        from tools.knowledge_updater import KnowledgeUpdater
        return KnowledgeUpdater(str(tmp_path / "brain.md"))

    def test_scoring_recency(self, tmp_path):
        from tools.knowledge_updater import PaperEntry
        ku = self._ku(tmp_path)
        import datetime as dt
        recent = PaperEntry("Streaming ASR latency", "A", dt.date.today().isoformat(),
                            "u", "low latency streaming asr tts", "arxiv")
        old = PaperEntry("Old topic", "B", "2000-01-01", "u2", "unrelated", "arxiv")
        assert ku._score(recent) > ku._score(old)

    def test_append_to_brain(self, tmp_path):
        from tools.knowledge_updater import PaperEntry
        ku = self._ku(tmp_path)
        p = PaperEntry("T", "A", "2025-01-01", "http://x", "abs", "arxiv", 0.5, "finding")
        ku._append_to_brain([p])
        assert "T" in (tmp_path / "brain.md").read_text(encoding="utf-8")

    def test_url_hash_stable(self):
        from tools.knowledge_updater import PaperEntry
        p1 = PaperEntry("T", "A", "2025", "http://X ", "a", "arxiv")
        p2 = PaperEntry("T2", "B", "2025", "http://x", "b", "arxiv")
        assert p1.url_hash == p2.url_hash


# --------------------------------------------------------------------------- #
# Integration (orchestrator end-to-end, all fallbacks)
# --------------------------------------------------------------------------- #
class TestIntegration:
    def _orch(self):
        from agent.orchestrator import VTuberOrchestrator
        cfg = {"memory": {"db_path": os.path.join(tempfile.mkdtemp(), "m.db")},
               "persona": {"name": "Aria", "description": "test persona"}}
        return VTuberOrchestrator(config=cfg, base_dir=tempfile.mkdtemp())

    def test_text_turn_end_to_end(self):
        out = self._orch().handle_turn(text="Hi, I'm so happy to be here!")
        assert "reply" in out and out["reply"]
        assert out["emotion"]
        assert "turn" in out["latencies_ms"]

    def test_empty_input_rejected(self):
        out = self._orch().handle_turn(text="   ")
        assert "error" in out

    def test_no_input_error(self):
        out = self._orch().handle_turn()
        assert "error" in out

    def test_propose_improvements(self):
        out = self._orch().propose_improvements()
        assert out["proposals"] and len(out["proposals"]) >= 3

    def test_benchmark_runs(self):
        orch = self._orch()
        res = orch.run_benchmark("t", [{"text": "hello", "reference": "hello"}])
        assert res["samples"] == 1

    def test_prometheus_metrics(self):
        orch = self._orch()
        orch.handle_turn(text="hello")
        assert "vtuber_turns_total" in orch.prometheus_metrics()


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #
class TestCLISmoke:
    def test_main_imports(self):
        import agent.main as m
        assert hasattr(m, "create_app")

    def test_config_loader(self):
        import agent.main as m
        cfg = m._load_config()
        assert isinstance(cfg, dict)

    def test_app_created(self):
        import agent.main as m
        # app may be None only if fastapi missing; in CI it should build
        assert m.app is None or hasattr(m.app, "routes")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
