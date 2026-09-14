"""Connect/detect/confirm/start/stop/estop -- control_loop.py's staged
rollout state machine (IDLE -> CONNECTED -> DETECTING -> AWAITING_CONFIRM
-> ARMED -> RUNNING) exposed over HTTP for the dashboard."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from franka_deploy.api import persist
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
        # Reuse the shared CameraManager (same one /api/cameras/start uses)
        # rather than opening the cameras a second time -- a RealSense
        # device can't be opened by two pipelines at once.
        if not APP_STATE.camera_manager.roles() and cfg.cameras:
            APP_STATE.camera_manager.start(build_cameras(cfg.cameras))
        safety_kwargs = dict(v_max=cfg.v_max, a_max=cfg.a_max, j_max=cfg.j_max, filter_wn=cfg.filter_wn)
        APP_STATE.loop = ControlLoop(cfg.robot_node_address, APP_STATE.camera_manager,
                                      cfg.request_spec, cfg.loop, safety_kwargs,
                                      workspace_bounds=APP_STATE.workspace)
    return APP_STATE.loop


@router.get("/workspace")
def get_workspace():
    return APP_STATE.workspace.to_dict()


@router.post("/workspace/add_point")
def add_workspace_point():
    """Records the robot's CURRENT measured EE position as one corner of
    the safety fence. Call this after hand-guiding the robot there
    (Programming/white mode on Desk) -- works fine while read_only, since
    it only reads state."""
    loop = _ensure_loop()
    if loop.state.value == "idle":
        raise HTTPException(400, "call /connect first -- no robot state to read yet")
    try:
        loop.add_workspace_point()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"add_point failed: {e}") from e
    persist.save_workspace(APP_STATE.workspace)
    return APP_STATE.workspace.to_dict()


@router.post("/workspace/remove_point")
def remove_workspace_point(body: dict):
    try:
        APP_STATE.workspace.remove_point(int(body["index"]))
    except IndexError as e:
        raise HTTPException(400, f"no point at index {body.get('index')}") from e
    persist.save_workspace(APP_STATE.workspace)
    return APP_STATE.workspace.to_dict()


@router.post("/workspace/clear")
def clear_workspace():
    APP_STATE.workspace.clear()
    persist.save_workspace(APP_STATE.workspace)
    return APP_STATE.workspace.to_dict()


@router.post("/workspace/margin")
def set_workspace_margin(body: dict):
    APP_STATE.workspace.margin = float(body["margin"])
    persist.save_workspace(APP_STATE.workspace)
    return APP_STATE.workspace.to_dict()


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
    # "stopped" is a valid restart point, not just "armed": the confirmed
    # ActionSpaceSpec (control_loop.py's self._adapter) survives stop() --
    # only /disconnect clears it -- so repeated Stop -> Start during one
    # session shouldn't force detect+confirm again each time.
    if loop.state.value not in ("armed", "stopped"):
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


@router.post("/disconnect")
def disconnect():
    """Fully release the session, robot connection included, and clear it
    so /api/config accepts changes again. /stop and /estop alone only halt
    the policy loop -- the request_spec is baked into the PolicyClient at
    ControlLoop construction time, so changing it requires a fresh
    ControlLoop, which means releasing the current FCI connection first
    (only one client at a time)."""
    if APP_STATE.loop is not None:
        APP_STATE.loop.estop()
        APP_STATE.loop = None
    return {"state": "idle"}


@router.get("/state")
def get_state():
    return {"state": APP_STATE.loop.state.value if APP_STATE.loop else "idle"}
