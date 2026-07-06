"""Frontend bridge: wire the AI-layer JSON into the live upstream Live2D frontend.

Phase 7 deliverable. The AI layer (this project) exposes ``POST /api/v1/turn``
returning a "speak" command JSON. The upstream Open-LLM-VTuber frontend is a
browser/Electron app driven over a WebSocket. This bridge closes the gap with a
**sidecar relay**: it speaks the upstream WebSocket protocol on one side and
calls the orchestrator turn loop on the other, so the upstream frontend can
consume the enhanced pipeline without any protocol changes.

Two modes:

  * ``serve``  - run an asyncio WebSocket server (default port 8002). Connected
                 frontends send ``{"type":"user_input","text":"..."}`` (or audio
                 path) and receive the upstream-compatible speak envelope.
  * ``push``   - connect to a running upstream backend WebSocket and forward a
                 pre-computed command (one-shot / scripted broadcasting).

The envelope emitted to the frontend follows the Open-LLM-VTuber Live2D message
shape (audio URL + expression + motion + viseme lip-sync timeline), which is
exactly the AI-layer command augmented with the upstream ``type`` field. The
mapping is configurable via ``message_template`` so it can track upstream
version changes without a code edit.

Requires the optional ``websockets`` package (added to requirements.txt). When
it is missing the bridge logs a clear error and refuses to start rather than
crashing the host.
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Set, Callable, Awaitable

logger = logging.getLogger("vtuber.frontend_bridge")

# Default upstream-compatible envelope. The AI-layer command is merged into
# this template; ``type`` overrides the AI-layer "speak" so the upstream
# frontend dispatches it correctly.
DEFAULT_MESSAGE_TEMPLATE: Dict[str, Any] = {
    "type": "avatar-speak",
    "audio": None,
    "text": "",
    "expression": "",
    "motion": "",
    "live2d_params": {},
    "lip_sync": [],
    "emotion": "neutral",
}


@dataclass
class BridgeConfig:
    host: str = "0.0.0.0"
    port: int = 8002
    path: str = "/vtuber"
    upstream_ws_url: Optional[str] = None  # for push mode
    message_template: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MESSAGE_TEMPLATE))
    ping_interval: float = 20.0
    ping_timeout: float = 20.0


class FrontendBridge:
    """Async WebSocket relay between the upstream Live2D frontend and the
    VTuber orchestrator turn loop."""

    def __init__(self, orchestrator, config: Optional[BridgeConfig] = None):
        self.orch = orchestrator
        self.config = config or BridgeConfig()
        self._clients: Set[Any] = set()
        self._server: Optional[Any] = None

    # -- envelope ------------------------------------------------------------
    def build_envelope(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Merge the AI-layer speak command into the upstream envelope."""
        command = turn.get("command") or {}
        envelope = dict(self.config.message_template)
        # pull the AI-layer fields forward, keeping the template's type unless
        # the AI-layer command's type is a richer upstream type
        envelope["type"] = envelope.get("type") or command.get("type", "avatar-speak")
        envelope["audio"] = command.get("audio")
        envelope["text"] = turn.get("reply") or command.get("text", "")
        envelope["expression"] = command.get("expression", "")
        envelope["motion"] = command.get("motion", "")
        envelope["live2d_params"] = command.get("live2d_params", {})
        envelope["lip_sync"] = command.get("lip_sync", [])
        envelope["emotion"] = command.get("emotion", turn.get("emotion", "neutral"))
        envelope["session_id"] = turn.get("session_id")
        envelope["latencies_ms"] = turn.get("latencies_ms", {})
        return envelope

    # -- serve mode ----------------------------------------------------------
    async def serve(self):
        try:
            import websockets
        except Exception as exc:  # noqa: BLE001
            logger.error("websockets package is required to run the frontend bridge: %s", exc)
            raise

        async def handler(ws):
            self._clients.add(ws)
            client_id = id(ws)
            logger.info("frontend client connected (%d clients)", len(self._clients))
            try:
                async for raw in ws:
                    msg = self._decode(raw)
                    if msg is None:
                        continue
                    response = await self._handle_message(msg)
                    if response is not None:
                        await ws.send(self._encode(response))
            except Exception as exc:  # noqa: BLE001 - per-client isolation
                logger.warning("client handler error: %s", exc)
            finally:
                self._clients.discard(ws)
                logger.info("frontend client disconnected (%d clients)", len(self._clients))

        logger.info("frontend bridge listening on ws://%s:%d%s",
                    self.config.host, self.config.port, self.config.path)
        self._server = await websockets.serve(
            handler, self.config.host, self.config.port,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
        )
        return self._server

    async def _handle_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        mtype = msg.get("type", "user_input")
        if mtype in ("user_input", "text", "chat"):
            text = (msg.get("text") or msg.get("message") or "").strip()
            if not text:
                return {"type": "error", "error": "empty text"}
            user_id = msg.get("user_id", "default")
            turn = await self._run_turn(text=text, user_id=user_id,
                                        session_id=msg.get("session_id"))
            if turn.get("error"):
                return {"type": "error", "error": turn["error"]}
            return self.build_envelope(turn)
        if mtype in ("audio_input", "audio"):
            audio_path = msg.get("audio") or msg.get("audio_path")
            if not audio_path:
                return {"type": "error", "error": "no audio path"}
            turn = await self._run_turn(audio_path=audio_path,
                                        user_id=msg.get("user_id", "default"),
                                        session_id=msg.get("session_id"))
            if turn.get("error"):
                return {"type": "error", "error": turn["error"]}
            return self.build_envelope(turn)
        if mtype == "ping":
            return {"type": "pong"}
        return {"type": "error", "error": f"unknown message type: {mtype}"}

    async def _run_turn(self, **kwargs) -> Dict[str, Any]:
        """Run the (sync) orchestrator turn in a worker thread so the asyncio
        loop stays responsive."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.orch.handle_turn(**kwargs))

    # -- broadcast / push ----------------------------------------------------
    async def broadcast(self, envelope: Dict[str, Any]):
        payload = self._encode(envelope)
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("broadcast to a client failed: %s", exc)
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def push_once(self, envelope: Dict[str, Any]):
        """Push a single envelope to a configured upstream backend WebSocket."""
        if not self.config.upstream_ws_url:
            raise ValueError("upstream_ws_url is required for push mode")
        import websockets
        async with websockets.connect(self.config.upstream_ws_url) as ws:
            await ws.send(self._encode(envelope))
            try:
                ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
                return self._decode(ack)
            except asyncio.TimeoutError:
                return None

    # -- codec helpers -------------------------------------------------------
    @staticmethod
    def _decode(raw) -> Optional[Dict[str, Any]]:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:  # noqa: BLE001
                return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        if isinstance(raw, dict):
            return raw
        return None

    @staticmethod
    def _encode(obj: Dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    # -- lifecycle -----------------------------------------------------------
    async def shutdown(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


def run_bridge(orchestrator, host: str = "0.0.0.0", port: int = 8002,
               path: str = "/vtuber", upstream_ws_url: Optional[str] = None):
    """Blocking entry point: start the bridge server and run until interrupted."""
    cfg = BridgeConfig(host=host, port=port, path=path, upstream_ws_url=upstream_ws_url)
    bridge = FrontendBridge(orchestrator, cfg)

    async def _main():
        await bridge.serve()
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await bridge.shutdown()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("frontend bridge stopped by interrupt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from scripts._common import build_orchestrator
    run_bridge(build_orchestrator())
