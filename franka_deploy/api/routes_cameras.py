"""Camera preview -- lets the operator confirm a camera is wired up and
framed correctly WITHOUT connecting to the robot, and independent of any
running control-loop session. Backed by cameras/manager.py's own
background thread per camera, shared with ControlLoop so a camera is
never opened twice.
"""

from __future__ import annotations

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from franka_deploy.api.state import APP_STATE

router = APIRouter(prefix="/api/cameras")


@router.get("/devices")
def list_devices():
    """USB device enumeration only -- no pipeline opened, safe to call
    anytime (even while another camera is mid-stream)."""
    from franka_deploy.cameras.realsense import list_devices as _list_devices

    try:
        return _list_devices()
    except Exception as e:  # noqa: BLE001 -- e.g. pyrealsense2 missing/no permission
        raise HTTPException(500, f"device scan failed: {e}") from e


@router.post("/start")
def start_preview():
    cfg = APP_STATE.config
    if not cfg.cameras:
        raise HTTPException(400, "no cameras configured -- POST /api/config first")
    try:
        APP_STATE.camera_manager.start(cfg.cameras)
    except RuntimeError as e:
        # Most common real cause: another process (a collection GUI, a
        # previous crashed session) already has the RealSense pipeline
        # open -- one process at a time per camera.
        raise HTTPException(
            409, f"camera open failed -- is another app (e.g. a data-collection GUI) "
                 f"already using it? ({e})"
        ) from e
    return {"roles": APP_STATE.camera_manager.roles()}


@router.post("/stop")
def stop_preview():
    if APP_STATE.loop is not None:
        raise HTTPException(409, "a session is using these cameras -- disconnect first")
    APP_STATE.camera_manager.stop()
    return {"roles": []}


@router.get("/{role}/frame.jpg")
def get_frame(role: str):
    rgb, error = APP_STATE.camera_manager.get_latest(role)
    if error is not None:
        raise HTTPException(500, error)
    if rgb is None:
        raise HTTPException(503, "no frame yet -- still starting, or /api/cameras/start not called")
    bgr = rgb[:, :, ::-1]
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise HTTPException(500, "jpeg encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
