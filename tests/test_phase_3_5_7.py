"""Tests for Phase 3 / 5 / 7 deliverables.

These run without any ML deps, GPU, API keys, or network: heavy models degrade
to deterministic fallbacks, the LLM client falls back to canned replies, and the
scripts operate on synthetic manifests. Run: pytest tests/test_phase_3_5_7.py
"""

import os
import sys
import json
import wave
import struct
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


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
# Phase 3 - HFModelManager ASR backend selection
# --------------------------------------------------------------------------- #
class TestASRBackend:
    def test_backend_routing_default(self):
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        from tools.hf_model_manager import VALID_ASR_BACKENDS
        assert mgr.asr_backend in VALID_ASR_BACKENDS and mgr.asr_backend

    def test_set_asr_backend_swaps_and_evicts(self):
        from tools.hf_model_manager import get_manager
        mgr = get_manager()
        original = mgr.asr_backend
        mgr.set_asr_backend("faster-whisper", compute="int8")
        assert mgr.asr_backend == "faster-whisper"
        assert mgr.faster_whisper_compute == "int8"
        # model cache should be evicted so the next load uses the new backend
        assert "whisper" not in mgr._models
        mgr.set_asr_backend(original)

    def test_set_asr_backend_rejects_unknown(self):
        from tools.hf_model_manager import get_manager
        mgr = get_manager()
        with pytest.raises(ValueError):
            mgr.set_asr_backend("nonsense")

    def test_transcribe_fallback_when_model_missing(self):
        from tools.hf_model_manager import get_manager
        mgr = get_manager()
        # force faster-whisper path that will fail to import -> graceful fallback
        mgr.set_asr_backend("faster-whisper")
        out = mgr.transcribe("/nonexistent/audio.wav")
        assert out["fallback"] is True and out["text"] == ""
        mgr.set_asr_backend("openai-whisper")


# --------------------------------------------------------------------------- #
# Phase 3 - validation scripts
# --------------------------------------------------------------------------- #
class TestValidateScripts:
    def test_load_manifest_jsonl(self, tmp_path):
        from scripts._common import load_manifest
        m = tmp_path / "m.jsonl"
        m.write_text('{"audio": "a.wav", "reference": "hello"}\n'
                     '{"audio": "b.wav", "reference": "world"}\n',
                     encoding="utf-8")
        rows = load_manifest(str(m))
        assert len(rows) == 2 and rows[0]["reference"] == "hello"
        # audio resolved relative to manifest dir
        assert os.path.isabs(rows[0]["audio"])

    def test_load_manifest_csv(self, tmp_path):
        from scripts._common import load_manifest
        m = tmp_path / "m.csv"
        m.write_text("audio,reference\na.wav,hello\nb.wav,world\n", encoding="utf-8")
        rows = load_manifest(str(m))
        assert rows[1]["reference"] == "world"

    def test_validate_asr_runs_with_fallbacks(self, tmp_path):
        from scripts.validate_asr import run
        m = tmp_path / "m.jsonl"
        m.write_text('{"audio": "missing1.wav", "reference": "hello world"}\n'
                      '{"audio": "missing2.wav", "reference": "goodbye"}\n',
                      encoding="utf-8")
        out = run(str(m), "test-asr", backend="openai-whisper",
                  report_path=str(tmp_path / "asr.md"))
        assert out["metrics"]["samples"] == 2
        assert out["metrics"]["fallbacks"] == 2
        assert out["metrics"]["asr_wer"] == 1.0  # empty transcripts vs reference
        assert os.path.exists(tmp_path / "asr.md")

    def test_validate_tts_runs_with_fallbacks(self, tmp_path):
        from scripts.validate_tts import run
        out = run(None, [{"text": "hello there", "emotion": "joy"},
                         {"text": "bye now", "emotion": "neutral"}],
                  "en", str(tmp_path / "tts"), "test-tts",
                  report_path=str(tmp_path / "tts.md"))
        assert out["metrics"]["samples"] == 2
        # without XTTS installed everything falls back to silence
        assert out["metrics"]["fallbacks"] == 2
        assert os.path.exists(tmp_path / "tts.md")

    def test_validate_diarization_fallback_without_token(self, tmp_path, monkeypatch):
        from scripts.validate_diarization import run
        monkeypatch.delenv("HF_TOKEN", raising=False)
        wav = tmp_path / "room.wav"
        _make_wav(str(wav), seconds=1.0)
        out = run(str(wav), None, "test-diar",
                  report_path=str(tmp_path / "dia.md"))
        assert out["metrics"]["fallback"] is True
        assert out["metrics"]["speakers"] == 1
        assert os.path.exists(tmp_path / "dia.md")

    def test_diarization_der_against_expected(self):
        from scripts.validate_diarization import diarization_error_rate
        ref = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
               {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}]
        hyp_perfect = list(ref)
        assert diarization_error_rate(ref, hyp_perfect) == 0.0
        # hypothesis mislabels the second turn -> nonzero DER
        hyp_bad = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
                   {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        assert diarization_error_rate(ref, hyp_bad) > 0.0


# --------------------------------------------------------------------------- #
# Phase 3/7 - apply proposal #1 gated re-benchmark
# --------------------------------------------------------------------------- #
class TestApplyProposalFasterWhisper:
    def test_gated_loop_runs_end_to_end(self, tmp_path):
        from scripts.apply_proposal_faster_whisper import run
        cases = [{"text": "hello there", "reference": "hello there"},
                 {"text": "good morning", "reference": "good morning"}]
        out = run(cases, "int8_float16", "b", "c",
                  str(tmp_path / "gate.md"), keep_candidate=False)
        assert "baseline" in out and "candidate" in out
        assert "passed" in out["gate"]
        assert os.path.exists(tmp_path / "gate.md")
        # keep_candidate=False restores the original backend
        from tools.hf_model_manager import get_manager
        assert get_manager().asr_backend == out["original_backend"]


# --------------------------------------------------------------------------- #
# Phase 5 - scheduler wiring
# --------------------------------------------------------------------------- #
class TestSchedulerWiring:
    def _orch(self):
        from agent.orchestrator import VTuberOrchestrator
        return VTuberOrchestrator(
            config={"memory": {"db_path": os.path.join(tempfile.mkdtemp(), "m.db")}},
            base_dir=tempfile.mkdtemp())

    def test_scheduler_status_before_start(self):
        orch = self._orch()
        s = orch.scheduler_status()
        assert s["running"] is False and s["jobs"] == []

    def test_start_scheduler_without_apscheduler_is_none(self, monkeypatch):
        orch = self._orch()
        # force APScheduler import to fail
        monkeypatch.setitem(__import__("sys").modules, "apscheduler.schedulers.background", None)
        assert orch.start_scheduler("daily") is None


# --------------------------------------------------------------------------- #
# Phase 7a - frontend bridge envelope
# --------------------------------------------------------------------------- #
class TestFrontendBridge:
    def _orch(self):
        from agent.orchestrator import VTuberOrchestrator
        return VTuberOrchestrator(
            config={"memory": {"db_path": os.path.join(tempfile.mkdtemp(), "m.db")}},
            base_dir=tempfile.mkdtemp())

    def test_build_envelope_shape(self):
        from agent.frontend_bridge import FrontendBridge, BridgeConfig
        orch = self._orch()
        turn = orch.handle_turn(text="I'm so happy today!")
        env = FrontendBridge(orch, BridgeConfig()).build_envelope(turn)
        assert env["type"] == "avatar-speak"
        assert env["text"] == turn["reply"]
        assert env["expression"] == turn["command"]["expression"]
        assert "lip_sync" in env and env["emotion"]

    def test_decode_encode_roundtrip(self):
        from agent.frontend_bridge import FrontendBridge
        obj = {"type": "ping", "n": 1}
        assert FrontendBridge._decode(FrontendBridge._encode(obj)) == obj
        assert FrontendBridge._decode(b"not json") is None


# --------------------------------------------------------------------------- #
# Phase 7c - proposal feeder
# --------------------------------------------------------------------------- #
class TestProposalFeeder:
    def test_feed_writes_and_dedups(self, tmp_path):
        from agent.proposal_feeder import ProposalFeeder
        feeder = ProposalFeeder(str(tmp_path))
        proposals = [
            {"title": "A", "citation": "https://arxiv.org/abs/1", "change": "x",
             "rationale": "y", "expected_impact": "+1", "risk": "low",
             "target_metric": "latency"},
            {"title": "B", "citation": "https://arxiv.org/abs/2", "change": "x2",
             "rationale": "y2", "expected_impact": "+2", "risk": "medium",
             "target_metric": "mos"},
        ]
        out = feeder.feed(proposals, {"asr_wer": 0.1})
        assert out["written"] == 2
        # feed again -> deduped to 0
        out2 = feeder.feed(proposals, {"asr_wer": 0.1})
        assert out2["written"] == 0
        # missing citation is dropped
        out3 = feeder.feed([{"title": "C", "citation": ""}], {})
        assert out3["written"] == 0
        assert len(feeder.read_feed()) == 2
        # check that the feed file exists and has content
        assert os.path.exists(feeder.feed_path)
        snap = feeder.latest_snapshot()
        assert snap is not None and snap["count"] == 2


# --------------------------------------------------------------------------- #
# Phase 7e - MOS rating collection + calibration
# --------------------------------------------------------------------------- #
class TestMosCalibration:
    def test_save_and_summary(self):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(os.path.join(tempfile.mkdtemp(), "m.db"))
        mm.save_mos_rating("/a.wav", 4.2, text="hi", emotion="joy", proxy_mos=4.0)
        mm.save_mos_rating("/b.wav", 3.8, text="oh", emotion="sadness", proxy_mos=4.1)
        s = mm.mos_summary()
        assert s["n"] == 2 and s["avg_human_mos"] is not None
        mm.close()

    def test_rating_out_of_range_rejected(self):
        from agent.memory.memory_manager import MemoryManager
        mm = MemoryManager(os.path.join(tempfile.mkdtemp(), "m.db"))
        with pytest.raises(ValueError):
            mm.save_mos_rating("/a.wav", 5.5)
        with pytest.raises(ValueError):
            mm.save_mos_rating("/a.wav", 0.5)
        mm.close()

    def test_least_squares(self):
        from scripts.calibrate_mos import _least_squares
        # y = 2x + 1 exactly
        xs = [1, 2, 3, 4]
        ys = [3, 5, 7, 9]
        slope, intercept, r2 = _least_squares(xs, ys)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)

    def test_calibrate_insufficient_then_calibrated(self, tmp_path, monkeypatch):
        from scripts import calibrate_mos
        from agent.orchestrator import VTuberOrchestrator

        orch = VTuberOrchestrator(
            config={"memory": {"db_path": os.path.join(tempfile.mkdtemp(), "m.db")}},
            base_dir=tempfile.mkdtemp())
        # patch the shared orchestrator builder used by the script
        monkeypatch.setattr(calibrate_mos, "build_orchestrator", lambda: orch)

        out = calibrate_mos.run("human", min_samples=5,
                                out_path=str(tmp_path / "cal.json"))
        assert out["status"] == "insufficient_data"

        # add enough ratings on a known linear relation: human = proxy + 0.5
        for px in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            orch.memory().save_mos_rating(f"/clip_{px}.wav", px + 0.5,
                                         emotion="joy", proxy_mos=px)
        out2 = calibrate_mos.run("human", min_samples=4,
                                 out_path=str(tmp_path / "cal2.json"))
        assert out2["status"] == "calibrated"
        assert out2["slope"] == pytest.approx(1.0, rel=1e-6)
        assert out2["intercept"] == pytest.approx(0.5, rel=1e-6)
        assert calibrate_mos.calibrated_mos(2.0, str(tmp_path / "cal2.json")) == pytest.approx(2.5)
        orch.memory().close()


# --------------------------------------------------------------------------- #
# Phase 7d - Prometheus exposition format
# --------------------------------------------------------------------------- #
class TestPrometheusMetrics:
    def _orch(self):
        from agent.orchestrator import VTuberOrchestrator
        return VTuberOrchestrator(
            config={"memory": {"db_path": os.path.join(tempfile.mkdtemp(), "m.db")}},
            base_dir=tempfile.mkdtemp())

    def test_metrics_text_format(self):
        orch = self._orch()
        orch.handle_turn(text="hello there")
        text = orch.prometheus_metrics()
        assert "# HELP vtuber_turns_total" in text
        assert "# TYPE vtuber_turns_total counter" in text
        assert "# TYPE vtuber_turn_latency_p95_ms gauge" in text
        assert "vtuber_turns_total 1" in text
        assert "vtuber_asr_wer" in text and "vtuber_human_mos" in text

    def test_stats_includes_scheduler_and_mos(self):
        orch = self._orch()
        s = orch.stats()
        assert "scheduler" in s and "gauges" in s and "human_mos" in s["gauges"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

