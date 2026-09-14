"""Tests for schema/apply.py's ActionSpaceAdapter -- converts one confirmed
-format response row into a joint target [8]. The EE cases exercise the
vendored FK/IK (kinematics/fr3_kinematics.py) and the new quat/rot6d
generalizations beyond that file's axis-angle-only convention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from franka_deploy.control_loop import _mat_to_quat  # noqa: E402
from franka_deploy.kinematics import fr3_kinematics as fk3  # noqa: E402
from franka_deploy.schema.apply import ActionSpaceAdapter  # noqa: E402
from franka_deploy.schema.spec import (  # noqa: E402
    ACTION_SPACE_EE_ABSOLUTE,
    ACTION_SPACE_EE_DELTA,
    ACTION_SPACE_JOINT_ABSOLUTE,
    ACTION_SPACE_JOINT_DELTA,
    ActionSpaceSpec,
)


def test_joint_absolute_passthrough():
    spec = ActionSpaceSpec(kind=ACTION_SPACE_JOINT_ABSOLUTE, dim=8, has_gripper=True, gripper_convention="0_1")
    adapter = ActionSpaceAdapter(spec)
    q_meas = np.zeros(7)
    row = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    out = adapter.row_to_joint_target(row, q_meas)
    assert np.allclose(out[:7], row[:7])
    assert out[7] == 0.8


def test_joint_delta_adds_to_measured():
    spec = ActionSpaceSpec(kind=ACTION_SPACE_JOINT_DELTA, dim=7, has_gripper=False, gripper_convention="none")
    adapter = ActionSpaceAdapter(spec)
    q_meas = np.full(7, 0.5)
    row = np.full(7, 0.01)
    out = adapter.row_to_joint_target(row, q_meas)
    assert np.allclose(out[:7], q_meas + 0.01)


def test_ee_absolute_quat_round_trips_through_ik():
    q_true = np.array([0.1, -0.3, 0.2, -2.0, 0.1, 2.0, 0.5])
    T = fk3.fk(q_true)
    pos, quat = T[:3, 3], _mat_to_quat(T[:3, :3])
    spec = ActionSpaceSpec(kind=ACTION_SPACE_EE_ABSOLUTE, dim=8, has_gripper=True,
                            gripper_convention="0_1", rotation_repr="quat")
    adapter = ActionSpaceAdapter(spec)
    row = np.concatenate([pos, quat, [0.7]])
    q_seed = q_true + 0.05  # a real control tick's seed is the latest measured joints, near the true solution
    out = adapter.row_to_joint_target(row, q_seed)
    T_check = fk3.fk(out[:7])
    assert np.allclose(T_check[:3, 3], pos, atol=1e-3)
    assert out[7] == 0.7


def test_ee_delta_axis_angle_matches_vendored_formula():
    q_meas = np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398])
    spec = ActionSpaceSpec(kind=ACTION_SPACE_EE_DELTA, dim=7, has_gripper=True,
                            gripper_convention="-1_1", rotation_repr="axis_angle",
                            delta_pos_scale=0.05, delta_rot_scale=0.5)
    adapter = ActionSpaceAdapter(spec)
    row = np.array([0.5, -0.2, 0.0, 0.1, 0.0, 0.0, 1.0])
    out = adapter.row_to_joint_target(row, q_meas)
    expected = fk3.ee_step_to_joint(row, q_meas, pos_max=0.05, rot_max=0.5)
    assert np.allclose(out[:7], expected[:7])
    assert out[7] == 1.0  # -1..1 convention: +1 -> fully closed -> 1.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
