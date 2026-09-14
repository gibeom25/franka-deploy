"""Connect/detect/confirm/start/stop/estop -- control_loop.py's staged
rollout state machine (IDLE -> CONNECTED -> DETECTING -> AWAITING_CONFIRM
-> ARMED -> RUNNING) exposed over HTTP for the dashboard."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from franka_deploy.api.state import APP_STATE
from franka_deploy.cameras import build_cameras
from franka_deploy.control_loop import ControlLoop
from franka_deploy.schema.spec import ActionSpaceSpec

router = APIRouter(prefix="/api/control")


def _ensure_loop() -> ControlLoop:
    cfg = APP_STATE.config
    if APP_STATE.loop is None:
        if cfg.request_spec is None:
            raise HTTPException(400, "no request_spec configured -- POST /api/config first")
        cameras = build_cameras(cfg.cameras)
        safety_kwargs = dict(v_max=cfg.v_max, a_max=cfg.a_max, j_max=cfg.j_max, filter_wn=cfg.filter_wn)
        APP_STATE.loop = ControlLoop(cfg.robot_ip, cameras, cfg.request_spec, cfg.loop, safety_kwargs)
    return APP_STATE.loop


@router.post("/connect")
def connect(body: dict = {}):
    loop = _ensure_loop()
    try:
        loop.connect(read_only=body.get("read_only", True))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"connect failed: {e}") from e
    return {"state": loop.state.value}


@router.post("/detect")
def detect(body: dict = {}):
    loop = _ensure_loop()
    if loop.state.value not in ("connected", "awaiting_confirm"):
        raise HTTPException(409, f"call /connect first (state={loop.state.value})")
    try:
        result = loop.detect(instruction=body.get("instruction"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"detect failed: {e}") from e
    return {
        "state": loop.state.value,
        "spec": asdict(result.spec),
        "sample_row": result.sample_row,
        "chunk_shape": result.chunk_shape,
        "candidates": [asdict(c) for c in result.candidates],
    }


@router.post("/confirm")
def confirm(body: dict):
    loop = _ensure_loop()
    loop.confirm(ActionSpaceSpec(**body["spec"]))
    return {"state": loop.state.value}


@router.post("/start")
def start(body: dict = {}):
    loop = _ensure_loop()
    if loop.state.value != "armed":
        raise HTTPException(409, f"call /detect then /confirm first (state={loop.state.value})")
    try:
        loop.start(instruction=body.get("instruction"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"start failed: {e}") from e
    return {"state": loop.state.value}


@router.post("/stop")
def stop():
    if APP_STATE.loop is not None:
        APP_STATE.loop.stop()
    return {"state": APP_STATE.loop.state.value if APP_STATE.loop else "idle"}


@router.post("/estop")
def estop():
    if APP_STATE.loop is not None:
        APP_STATE.loop.estop()
    return {"state": APP_STATE.loop.state.value if APP_STATE.loop else "idle"}


@router.get("/state")
def get_state():
    return {"state": APP_STATE.loop.state.value if APP_STATE.loop else "idle"}
