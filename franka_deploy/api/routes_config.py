"""Config CRUD: connection/request schema, safety limits, camera roles.

Raw-dict request bodies rather than pydantic models -- this is a
single-operator local tool, not a public API, and the schema dataclasses
(schema/spec.py) already reject unknown/missing fields on construction.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from franka_deploy.api.state import APP_STATE
from franka_deploy.cameras import CameraConfig
from franka_deploy.control_loop import LoopConfig
from franka_deploy.schema.serialize import request_spec_from_dict, request_spec_to_dict

router = APIRouter(prefix="/api/config")

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "configs" / "examples"


@router.get("")
def get_config():
    cfg = APP_STATE.config
    return {
        "robot_ip": cfg.robot_ip,
        "cameras": [asdict(c) for c in cfg.cameras],
        "request_spec": request_spec_to_dict(cfg.request_spec) if cfg.request_spec else None,
        "loop": asdict(cfg.loop),
        "v_max": cfg.v_max, "a_max": cfg.a_max, "j_max": cfg.j_max, "filter_wn": cfg.filter_wn,
    }


@router.post("")
def set_config(body: dict):
    if APP_STATE.loop is not None:
        raise HTTPException(409, "stop the running session before changing config")
    cfg = APP_STATE.config
    if "robot_ip" in body:
        cfg.robot_ip = body["robot_ip"]
    if "cameras" in body:
        cfg.cameras = [CameraConfig(**c) for c in body["cameras"]]
    if "request_spec" in body and body["request_spec"] is not None:
        cfg.request_spec = request_spec_from_dict(body["request_spec"])
    if "loop" in body:
        cfg.loop = LoopConfig(**body["loop"])
    for k in ("v_max", "a_max", "j_max", "filter_wn"):
        if k in body:
            setattr(cfg, k, body[k])
    return get_config()


@router.get("/examples")
def list_examples():
    return sorted(p.stem for p in EXAMPLES_DIR.glob("*.yaml"))


@router.get("/examples/{name}")
def get_example(name: str):
    path = EXAMPLES_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(404, f"no example named {name!r}")
    return yaml.safe_load(path.read_text())
