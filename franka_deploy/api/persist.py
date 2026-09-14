"""Disk persistence for AppConfig + WorkspaceBounds.

Previously both lived only in process memory -- any app-server restart
silently wiped everything the operator had typed in, including a
workspace fence that took real hand-guiding effort to record. Saved to
``~/.franka-deploy/`` (matches run_franka_deploy.sh's log directory
convention), not inside the git-tracked package -- this is live,
machine-specific operator state, not project source.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from franka_deploy.cameras import CameraConfig
from franka_deploy.control_loop import LoopConfig
from franka_deploy.schema.serialize import request_spec_from_dict, request_spec_to_dict
from franka_deploy.safety.workspace_bounds import WorkspaceBounds

STATE_DIR = Path.home() / ".franka-deploy"
CONFIG_PATH = STATE_DIR / "config.json"
WORKSPACE_PATH = STATE_DIR / "workspace.json"


def save_config(cfg) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "robot_node_address": cfg.robot_node_address,
        "cameras": [asdict(c) for c in cfg.cameras],
        "request_spec": request_spec_to_dict(cfg.request_spec) if cfg.request_spec else None,
        "loop": asdict(cfg.loop),
        "v_max": cfg.v_max, "a_max": cfg.a_max, "j_max": cfg.j_max, "filter_wn": cfg.filter_wn,
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def load_config_into(cfg) -> bool:
    """Mutates cfg in place from the last save, if any. Returns whether a
    saved file existed (so the caller can log it, not required to act on)."""
    if not CONFIG_PATH.exists():
        return False
    data = json.loads(CONFIG_PATH.read_text())
    cfg.robot_node_address = data.get("robot_node_address", cfg.robot_node_address)
    if "cameras" in data:
        cfg.cameras = [CameraConfig(**c) for c in data["cameras"]]
    if data.get("request_spec"):
        cfg.request_spec = request_spec_from_dict(data["request_spec"])
    if data.get("loop"):
        cfg.loop = LoopConfig(**data["loop"])
    for k in ("v_max", "a_max", "j_max", "filter_wn"):
        if k in data:
            setattr(cfg, k, data[k])
    return True


def save_workspace(ws: WorkspaceBounds) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_PATH.write_text(json.dumps({"points": ws.points, "margin": ws.margin}, indent=2))


def load_workspace_into(ws: WorkspaceBounds) -> bool:
    if not WORKSPACE_PATH.exists():
        return False
    data = json.loads(WORKSPACE_PATH.read_text())
    ws.points = data.get("points", [])
    ws.margin = data.get("margin", ws.margin)
    return True
