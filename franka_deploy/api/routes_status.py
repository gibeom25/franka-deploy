"""WebSocket telemetry stream for the dashboard (~10 Hz): joint state,
policy latency, replan/late counters, current session state."""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from franka_deploy.api.state import APP_STATE

router = APIRouter()


@router.websocket("/ws/telemetry")
async def telemetry_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            loop = APP_STATE.loop
            if loop is None:
                payload = {"state": "idle"}
            else:
                t = loop.get_telemetry()
                payload = asdict(t)
                # loop.state is the authoritative live state (connect/detect/
                # confirm all update it); t.state is a snapshot only ever
                # refreshed inside _run()'s tick loop (RUNNING/ERROR), so it
                # was stuck reporting a stale "idle" through every earlier
                # stage -- CONNECTED, DETECTING, AWAITING_CONFIRM, ARMED --
                # right after the corresponding button's own response had
                # already shown the correct state for a moment.
                payload["state"] = loop.state.value
            await ws.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
