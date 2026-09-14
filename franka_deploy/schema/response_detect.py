"""Auto-detects what one action-chunk row means: joint_absolute / joint_delta /
ee_absolute / ee_delta, dimensionality, and gripper presence/convention.

Used once per session right after the first ``/predict`` response. Its
result is always shown to the user for a one-click confirm before any
motion is commanded (control_loop.py) -- this heuristic only has to be a
good first guess, not perfect. Whatever it picks, the reference filter
(robot/reference_filter.py) still clamps velocity/accel/jerk before
anything reaches the robot, so a wrong guess produces a bounded-rate motion
in the wrong direction, never an unbounded one.

The two families are told apart by magnitude, not just shape, mirroring the
precedent already validated in
``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py``
(``ee_mode = chunk.shape[1] == 7``) -- this generalizes that single
hardcoded check into a scored heuristic over all four action spaces.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from franka_deploy.safety.limits import FR3_Q_LOWER, FR3_Q_UPPER
from franka_deploy.schema.spec import (
    ACTION_SPACE_EE_ABSOLUTE,
    ACTION_SPACE_EE_DELTA,
    ACTION_SPACE_JOINT_ABSOLUTE,
    ACTION_SPACE_JOINT_DELTA,
    ActionSpaceSpec,
    DetectionResult,
)

# Loose margins -- this only needs to separate "plausible joint angle near
# where the arm actually is" from "small delta" or "meters", not police
# exact limits (clip_to_joint_limits does that later, unconditionally).
_JOINT_RANGE_MARGIN = 0.3            # rad, beyond FR3_Q_LOWER/UPPER
_JOINT_CLOSE_TO_CURRENT = 1.0        # rad, "close enough to q_now to be absolute"
_SMALL_DELTA_JOINT = 0.3             # rad, per-step joint delta ceiling
_EE_WORKSPACE_RADIUS = (0.10, 0.95)  # meters, base-frame reach envelope
_SMALL_DELTA_POS = 0.15              # meters, per-step EE delta ceiling
_GRIPPER_RANGE_01 = (-0.05, 1.05)
_GRIPPER_RANGE_PM1 = (-1.05, 1.05)
_ROTATION_REPR = {3: "axis_angle", 4: "quat", 6: "rot6d"}


def _score_joint(row: np.ndarray, q_now: np.ndarray) -> tuple[str, float]:
    j = row[:7]
    # NOTE: "in range" only gates the *absolute* hypothesis. A delta is
    # judged purely by magnitude -- some joints' absolute range doesn't
    # even include zero (J6: [0.54, 4.52] rad), so a small delta near zero
    # can legitimately fail the absolute-range check while still obviously
    # being a delta, not a wildly-out-of-range absolute target.
    in_absolute_range = bool(
        np.all(j >= FR3_Q_LOWER - _JOINT_RANGE_MARGIN) and np.all(j <= FR3_Q_UPPER + _JOINT_RANGE_MARGIN)
    )
    max_dev = float(np.max(np.abs(j - q_now)))
    max_mag = float(np.max(np.abs(j)))
    if in_absolute_range and max_dev < _JOINT_CLOSE_TO_CURRENT:
        return ACTION_SPACE_JOINT_ABSOLUTE, max(0.3, 1.0 - max_dev / _JOINT_CLOSE_TO_CURRENT)
    if max_mag < _SMALL_DELTA_JOINT:
        return ACTION_SPACE_JOINT_DELTA, max(0.3, 1.0 - max_mag / _SMALL_DELTA_JOINT)
    if in_absolute_range:
        return ACTION_SPACE_JOINT_ABSOLUTE, 0.1
    return ACTION_SPACE_JOINT_DELTA, 0.0


def _score_ee(row: np.ndarray, ee_pos_now: np.ndarray) -> tuple[str, float]:
    pos = row[:3]
    radius = float(np.linalg.norm(pos))
    lo, hi = _EE_WORKSPACE_RADIUS
    if lo <= radius <= hi:
        dev = float(np.linalg.norm(pos - ee_pos_now))
        return ACTION_SPACE_EE_ABSOLUTE, max(0.3, 1.0 - min(dev, 1.0))
    if radius < _SMALL_DELTA_POS:
        return ACTION_SPACE_EE_DELTA, max(0.3, 1.0 - radius / _SMALL_DELTA_POS)
    return ACTION_SPACE_EE_ABSOLUTE, 0.05


def _gripper_convention(col: np.ndarray) -> str:
    lo, hi = float(col.min()), float(col.max())
    if _GRIPPER_RANGE_01[0] <= lo and hi <= _GRIPPER_RANGE_01[1]:
        return "0_1"
    if _GRIPPER_RANGE_PM1[0] <= lo and hi <= _GRIPPER_RANGE_PM1[1]:
        return "-1_1"
    return "0_1"  # unclear -- low confidence is what actually flags this upstream


def detect_action_space(chunk: np.ndarray, q_now: np.ndarray, ee_pos_now: np.ndarray) -> DetectionResult:
    """chunk: [T, D] raw response array (T>=1). q_now: [7] rad, measured at
    the same tick the observation for this chunk was taken. ee_pos_now: [3]
    meters, base frame, same tick."""
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 2 or chunk.shape[0] < 1:
        raise ValueError(f"expected a [T, D] response chunk with T>=1, got shape {chunk.shape}")
    T, D = chunk.shape
    row = chunk[0]

    candidates: list[ActionSpaceSpec] = []

    for has_gripper, joint_dim in ((True, 8), (False, 7)):
        if D != joint_dim:
            continue
        kind, conf = _score_joint(row, q_now)
        conv = _gripper_convention(chunk[:, -1]) if has_gripper else "none"
        candidates.append(ActionSpaceSpec(
            kind=kind, dim=D, has_gripper=has_gripper, gripper_convention=conv,
            rotation_repr=None, confidence=conf,
            notes=f"joint-shaped ({'with' if has_gripper else 'without'} gripper column)",
        ))

    for has_gripper in (True, False):
        for rot_dim in (3, 4, 6):
            expect_d = 3 + rot_dim + (1 if has_gripper else 0)
            if D != expect_d:
                continue
            kind, conf = _score_ee(row, ee_pos_now)
            conv = _gripper_convention(chunk[:, -1]) if has_gripper else "none"
            candidates.append(ActionSpaceSpec(
                kind=kind, dim=D, has_gripper=has_gripper, gripper_convention=conv,
                rotation_repr=_ROTATION_REPR[rot_dim], confidence=conf,
                notes=f"EE-shaped, pos3+{_ROTATION_REPR[rot_dim]}{'+gripper' if has_gripper else ''}",
            ))

    if not candidates:
        candidates.append(ActionSpaceSpec(
            kind=ACTION_SPACE_JOINT_ABSOLUTE, dim=D, has_gripper=False,
            gripper_convention="none", confidence=0.0,
            notes=f"no known action-space shape matches D={D}; defaulting to a "
                  f"joint_absolute guess over the first 7 columns -- confirm or "
                  f"override manually before starting.",
        ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return DetectionResult(spec=candidates[0], sample_row=row.tolist(),
                            chunk_shape=(T, D), candidates=candidates)
