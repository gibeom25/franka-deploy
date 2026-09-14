"""Resolves a RequestFieldSpec.source string to an actual value.

Three things a source can point at, matching the ``<kind>:<detail>`` syntax
in spec.py:
  camera:<role>   -> latest RGB frame (H,W,3) uint8 for that camera role
  robot:<key>     -> one entry of the current robot-state dict (see
                     ``robot_state_dict`` below for what keys exist)
  static:<value>  -> the literal string/number typed into the schema editor
  custom:<path>   -> a user plugin function (schema/plugins.py), called with
                     the same (cameras, robot_state) context as everything
                     else, for anything a generic pipeline can't produce
                     (e.g. a language embedding).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

from franka_deploy.schema.plugins import load_plugin


class SourceContext:
    """One snapshot of everything a request can pull fields from."""

    def __init__(self, cameras: dict[str, np.ndarray], robot_state: dict[str, Any]):
        self.cameras = cameras
        self.robot_state = robot_state


def robot_state_dict(q: np.ndarray, dq: np.ndarray, ee_pos: np.ndarray,
                      ee_quat: np.ndarray, gripper: float) -> dict[str, Any]:
    """Build the dict SourceContext.robot_state expects, from an FR3Runtime
    get_state() result (already split into pos/quat by the caller -- see
    control_loop.py). Keys a ``robot:<key>`` source may reference:
      joint_positions   -> [8] float, 7 rad + gripper 0..1
      joint_positions7  -> [7] float, rad only, no gripper
      joint_velocities  -> [7] float, rad/s
      ee_pos            -> [3] float, meters, base frame
      ee_quat           -> [4] float, xyzw
      ee_pos_quat       -> [7] float, pos+quat
      gripper           -> scalar 0..1
    """
    return {
        "joint_positions": np.append(q, gripper),
        "joint_positions7": q,
        "joint_velocities": dq,
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
        "gripper": gripper,
    }


def resolve_source(source: str, ctx: SourceContext) -> Any:
    if ":" not in source:
        raise ValueError(f"malformed source {source!r}, expected '<kind>:<detail>'")
    kind, detail = source.split(":", 1)

    if kind == "camera":
        if detail not in ctx.cameras:
            raise KeyError(f"camera role {detail!r} not configured (have {list(ctx.cameras)})")
        return ctx.cameras[detail]

    if kind == "robot":
        if detail not in ctx.robot_state:
            raise KeyError(f"robot state key {detail!r} not available (have {list(ctx.robot_state)})")
        return ctx.robot_state[detail]

    if kind == "static":
        # "static:1" -> int 1, "static:true" -> bool True, etc. -- lets a
        # static field carry a protocol constant (e.g. protocol_version=1)
        # correctly typed. Falls back to the raw string for ordinary text
        # ("static:pick up the cup" isn't valid JSON, so json.loads raises).
        try:
            return json.loads(detail)
        except (json.JSONDecodeError, ValueError):
            return detail

    if kind == "custom":
        fn: Callable[[SourceContext], Any] = load_plugin(detail)
        return fn(ctx)

    raise ValueError(f"unknown source kind {kind!r} in {source!r}")
