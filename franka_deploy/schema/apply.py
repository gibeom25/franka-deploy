"""Converts one confirmed-format action-chunk row into a joint target [8]
(7 rad + gripper 0..1) ready for the safety pipeline (EMASmoother ->
clip_to_joint_limits -> FR3Runtime.set_joint_target, see control_loop.py).

joint_absolute / joint_delta are near-passthrough. ee_absolute / ee_delta go
through the vendored analytic FK/IK (kinematics/fr3_kinematics.py) -- the
same math already validated live for EE-delta checkpoints in
~/teleop-franka/manipulation-stack-dev, generalized here to EE_absolute
targets and to rotation representations (quat, rot6d) beyond the
axis-angle-delta convention that repo hardcodes.
"""

from __future__ import annotations

import numpy as np

from franka_deploy.kinematics import fr3_kinematics as fk3
from franka_deploy.kinematics.rotations import quat_to_rot, rot6d_to_rot
from franka_deploy.schema.spec import (
    ACTION_SPACE_EE_ABSOLUTE,
    ACTION_SPACE_EE_DELTA,
    ACTION_SPACE_JOINT_ABSOLUTE,
    ACTION_SPACE_JOINT_DELTA,
    ActionSpaceSpec,
)

_ROT_DIM = {"axis_angle": 3, "quat": 4, "rot6d": 6, None: 0}


def _gripper_to_0_1(value: float, convention: str) -> float:
    if convention == "-1_1":
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _rotation_matrix(vec: np.ndarray, repr_kind: str) -> np.ndarray:
    if repr_kind == "quat":
        return quat_to_rot(vec)
    if repr_kind == "rot6d":
        return rot6d_to_rot(vec)
    if repr_kind == "axis_angle":
        return fk3.axis_angle_to_rot(vec)
    raise ValueError(f"unknown rotation_repr {repr_kind!r}")


class ActionSpaceAdapter:
    def __init__(self, spec: ActionSpaceSpec):
        self.spec = spec
        self._rot_dim = _ROT_DIM[spec.rotation_repr]

    def row_to_joint_target(self, row: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
        """row: one action-chunk row, as returned by the server (untouched).
        q_meas: [7] measured joints (rad) at the SAME tick this row is being
        applied -- delta modes re-anchor to this every call, matching the
        validated manipulation-stack convention (tracking lag never
        accumulates because the anchor is always the latest measurement,
        not the previous target)."""
        row = np.asarray(row, dtype=float)
        q_meas = np.asarray(q_meas, dtype=float)
        gripper = (
            _gripper_to_0_1(row[self.spec.dim - 1], self.spec.gripper_convention)
            if self.spec.has_gripper else 0.0
        )

        if self.spec.kind == ACTION_SPACE_JOINT_ABSOLUTE:
            q = row[:7]
        elif self.spec.kind == ACTION_SPACE_JOINT_DELTA:
            q = q_meas + row[:7] * self.spec.delta_joint_scale
        elif self.spec.kind == ACTION_SPACE_EE_ABSOLUTE:
            q = self._ee_absolute(row, q_meas)
        elif self.spec.kind == ACTION_SPACE_EE_DELTA:
            q = self._ee_delta(row, q_meas)
        else:
            raise ValueError(f"unhandled action space kind {self.spec.kind!r}")

        out = np.zeros(8)
        out[:7] = q
        out[7] = gripper
        return out

    def _ee_absolute(self, row: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
        pos = row[:3]
        rot = _rotation_matrix(row[3:3 + self._rot_dim], self.spec.rotation_repr)
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = rot
        return fk3.ik(T, q_meas)

    def _ee_delta(self, row: np.ndarray, q_meas: np.ndarray) -> np.ndarray:
        if self.spec.rotation_repr == "axis_angle":
            # Byte-identical to the manipulation-stack-validated formula --
            # reuse it directly rather than re-deriving the same math.
            delta7 = np.concatenate([row[:3], row[3:6], [0.0]])
            return fk3.ee_step_to_joint(
                delta7, q_meas,
                pos_max=self.spec.delta_pos_scale, rot_max=self.spec.delta_rot_scale,
            )[:7]
        # quat/rot6d delta: same re-anchor-to-measured-pose convention,
        # generalized from a normalized axis-angle signal to a full
        # rotation delta.
        T_now = fk3.fk(q_meas)
        R_now, p_now = T_now[:3, :3], T_now[:3, 3]
        dpos = row[:3] * self.spec.delta_pos_scale
        R_delta = _rotation_matrix(row[3:3 + self._rot_dim], self.spec.rotation_repr)
        T_tgt = np.eye(4)
        T_tgt[:3, 3] = p_now + R_now @ dpos
        T_tgt[:3, :3] = R_delta @ R_now
        return fk3.ik(T_tgt, q_meas)
