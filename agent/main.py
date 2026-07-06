"""main.py - entry point for open-llm-vtuber-enhanced.

Two surfaces over the same VTuberOrchestrator:
  * Click CLI    - chat / listen / benchmark / propose / update-knowledge /
                   feed-proposals / rate-mos / calibrate-mos / bridge / stats
  * FastAPI app  - REST endpoints for the Open-LLM-VTuber frontend / sidecar use

Run the CLI:        python -m agent.main chat
Run the server:     python -m agent.main serve   (or: uvicorn agent.main:app)
Run the bridge:     python -m agent.main bridge
"""

from __future__ import annotations

import os
import sys
import json
import logging

# make project root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vtuber.main")


def _load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config", "agent_config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("config load failed (%s); using defaults", exc)
    return {}


def _orchestrator():
    from agent.orchestrator import VTuberOrchestrator
    return VTuberOrchestrator(config=_load_config(), base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
try:
    import click

    @click.group()
    def cli():
        """open-llm-vtuber-enhanced - AI VTuber + research optimization agent."""

    @cli.command()
    @click.option("--user", default="default", help="user/viewer id")
    def chat(user):
        """Interactive text chat with the VTuber persona."""
        orch = _orchestrator()
        click.echo(f"[{orch.persona_name}] Hi! Type 'quit' to exit.")
        session = None
        while True:
            try:
                msg = click.prompt("you", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                break
            if msg.strip().lower() in ("quit", "exit"):
                break
            out = orch.handle_turn(text=msg, user_id=user, session_id=session)
            session = out.get("session_id", session)
            if out.get("error"):
                click.echo(f"  ! {out['error']}")
                continue
            click.echo(f"[{orch.persona_name}] {out['reply']}")
            click.echo(f"  (emotion={out['emotion']} latency={out['latencies_ms'].get('turn')}ms)")

    @cli.command()
    @click.option("--seconds", default=5.0, help="record duration")
    @click.option("--user", default="default")
    def listen(seconds, user):
        """Capture one microphone turn and respond."""
        orch = _orchestrator()
        ingest = orch.ingestor().capture_microphone(seconds=seconds)
        if not ingest.valid:
            click.echo(f"capture rejected: {ingest.reason}")
            return
        out = orch.handle_turn(audio_path=ingest.audio_path, user_id=user)
        click.echo(json.dumps(out, indent=2, default=str))

    @cli.command("update-knowledge")
    def update_knowledge():
        """Run the research-paper crawler once and append to the knowledge base."""
        orch = _orchestrator()
        click.echo(json.dumps(orch.update_knowledge(), indent=2))

    @cli.command()
    def propose():
        """Generate cited optimization proposals from the latest research."""
        orch = _orchestrator()
        click.echo(json.dumps(orch.propose_improvements(), indent=2))

    @cli.command("feed-proposals")
    @click.option("--out", "out_dir", default=None,
                  help="optional override for the proposal feed directory")
    def feed_proposals(out_dir):
        """Feed the latest proposals into the academic-research-enhanced feed."""
        from agent.proposal_feeder import ProposalFeeder
        orch = _orchestrator()
        proposals = orch.propose_improvements()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feeder = ProposalFeeder(base_dir)
        if out_dir:
            feeder.feed_dir = out_dir
            feeder.feed_path = os.path.join(out_dir, "feed.jsonl")
            os.makedirs(out_dir, exist_ok=True)
        result = feeder.feed(proposals["proposals"], proposals["based_on"])
        click.echo(json.dumps(result, indent=2, default=str))

    @cli.command()
    @click.option("--label", default="cli-run")
    def benchmark(label):
        """Run a small built-in benchmark suite over the live pipeline."""
        orch = _orchestrator()
        cases = [
            {"text": "Hello, how are you today?", "reference": "Hello, how are you today?"},
            {"text": "Tell me a fun fact about space.", "reference": "Tell me a fun fact about space."},
            {"text": "I'm feeling a little down.", "reference": "I'm feeling a little down."},
        ]
        click.echo(json.dumps(orch.run_benchmark(label, cases), indent=2, default=str))

    @cli.command()
    def stats():
        """Show agent stats and cost summary."""
        orch = _orchestrator()
        click.echo(json.dumps(orch.stats(), indent=2, default=str))

    @cli.command("rate-mos")
    @click.option("--audio", required=True, help="path to the synthesized clip")
    @click.option("--rating", required=True, type=float, help="human MOS rating 1.0-5.0")
    @click.option("--text", default=None, help="prompt that produced the clip")
    @click.option("--emotion", default=None)
    @click.option("--proxy", default=None, type=float, help="objective proxy MOS at rating time")
    @click.option("--rater", default="human")
    def rate_mos(audio, rating, text, emotion, proxy, rater):
        """Submit a human MOS rating for a synthesized clip."""
        orch = _orchestrator()
        rid = orch.memory().save_mos_rating(audio, rating, text=text, emotion=emotion,
                                           rater_id=rater, proxy_mos=proxy)
        click.echo(json.dumps({"id": rid, "rating": rating, "audio": audio}))

    @cli.command("calibrate-mos")
    @click.option("--min-samples", default=10, type=int)
    @click.option("--rater-id", default="human")
    def calibrate_mos(min_samples, rater_id):
        """Fit the objective MOS proxy against collected human ratings."""
        from scripts.calibrate_mos import run
        cal = run(rater_id, min_samples)
        click.echo(json.dumps(cal, indent=2))

    @cli.command()
    @click.option("--host", default="0.0.0.0")
    @click.option("--port", default=8001)
    @click.option("--schedule/--no-schedule", default=True,
                  help="start the in-process knowledge crawler scheduler on boot")
    def serve(host, port, schedule):
        """Start the FastAPI server."""
        import uvicorn
        os.environ["VTUBER_START_SCHEDULER"] = "1" if schedule else "0"
        uvicorn.run("agent.main:app", host=host, port=port, reload=False)

    @cli.command()
    @click.option("--host", default="0.0.0.0")
    @click.option("--port", default=8002)
    @click.option("--path", "ws_path", default="/vtuber")
    @click.option("--upstream", default=None,
                  help="optional upstream backend WebSocket URL for push mode")
    def bridge(host, port, ws_path, upstream):
        """Run the upstream Live2D frontend WebSocket bridge."""
        from agent.frontend_bridge import run_bridge
        orch = _orchestrator()
        run_bridge(orch, host=host, port=port, path=ws_path, upstream_ws_url=upstream)

except ImportError:  # click not installed
    cli = None


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel
    from typing import Optional, List

    app = FastAPI(title="open-llm-vtuber-enhanced",
                  description="AI VTuber pipeline + research optimization agent",
                  version="1.0.0")
    orch = _orchestrator()

    class TurnRequest(BaseModel):
        text: Optional[str] = None
        audio_path: Optional[str] = None
        user_id: str = "default"
        session_id: Optional[str] = None

    class BenchRequest(BaseModel):
        label: str = "api-run"
        cases: List[dict]

    class MosRatingRequest(BaseModel):
        audio: str
        rating: float
        text: Optional[str] = None
        emotion: Optional[str] = None
        proxy_mos: Optional[float] = None
        rater_id: str = "human"
        notes: str = ""

    @app.on_event("startup")
    def _startup():
        schedule = os.getenv("VTUBER_START_SCHEDULER", "1").lower() not in ("0", "false", "no")
        cron = (orch.config.get("knowledge_updater", {}) or {}).get("schedule", "daily")
        if schedule:
            try:
                orch.start_scheduler(cron=cron)
                logger.info("knowledge scheduler started on startup (cron=%s)", cron)
            except Exception as exc:  # noqa: BLE001
                logger.warning("scheduler startup failed (crawls still runnable on demand): %s", exc)

    @app.on_event("shutdown")
    def _shutdown():
        sched = orch._scheduler
        if sched is not None:
            try:
                sched.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass

    @app.get("/health")
    def health():
        return {"status": "ok", "persona": orch.persona_name}

    @app.post("/api/v1/turn")
    def turn(req: TurnRequest):
        if not req.text and not req.audio_path:
            raise HTTPException(400, "provide text or audio_path")
        out = orch.handle_turn(text=req.text, audio_path=req.audio_path,
                               user_id=req.user_id, session_id=req.session_id)
        if out.get("error"):
            raise HTTPException(422, out["error"])
        return out

    @app.post("/api/v1/knowledge/update")
    def knowledge_update():
        return orch.update_knowledge()

    @app.get("/api/v1/knowledge/schedule")
    def knowledge_schedule():
        return orch.scheduler_status()

    @app.get("/api/v1/proposals")
    def proposals():
        return orch.propose_improvements()

    @app.post("/api/v1/proposals/feed")
    def proposals_feed():
        from agent.proposal_feeder import ProposalFeeder
        prop = orch.propose_improvements()
        feeder = ProposalFeeder(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return feeder.feed(prop["proposals"], prop["based_on"])

    @app.post("/api/v1/benchmark")
    def benchmark(req: BenchRequest):
        return orch.run_benchmark(req.label, req.cases)

    @app.get("/api/v1/stats")
    def stats():
        return orch.stats()

    @app.post("/api/v1/mos/rate")
    def mos_rate(req: MosRatingRequest):
        if not (1.0 <= req.rating <= 5.0):
            raise HTTPException(400, "rating must be between 1.0 and 5.0")
        rid = orch.memory().save_mos_rating(
            req.audio, req.rating, text=req.text, emotion=req.emotion,
            rater_id=req.rater_id, proxy_mos=req.proxy_mos, notes=req.notes)
        return {"id": rid, "rating": req.rating, "audio": req.audio}

    @app.get("/api/v1/mos/summary")
    def mos_summary():
        return orch.memory().mos_summary()

    @app.post("/api/v1/mos/calibrate")
    def mos_calibrate(min_samples: int = 10, rater_id: str = "human"):
        from scripts.calibrate_mos import run
        return run(rater_id, min_samples)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return orch.prometheus_metrics()

    return app


# module-level app for `uvicorn agent.main:app`
try:
    app = create_app()
except Exception as exc:  # noqa: BLE001
    logger.warning("FastAPI app not created at import (%s)", exc)
    app = None


if __name__ == "__main__":
    if cli is not None:
        cli()
    else:
        print("Click not installed. Install requirements or run: uvicorn agent.main:app")
