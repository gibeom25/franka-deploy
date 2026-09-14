"""Synthetic-array tests for the action-space auto-detection heuristic --
no hardware or network needed. Mirrors the four action spaces the project
must support generally, generalizing the single hardcoded
``chunk.shape[1] == 7`` check already validated in
``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from franka_deploy.schema.response_detect import detect_action_space  # noqa: E402
from franka_deploy.schema.spec import (  # noqa: E402
    ACTION_SPACE_EE_ABSOLUTE,
    ACTION_SPACE_EE_DELTA,
    ACTION_SPACE_JOINT_ABSOLUTE,
    ACTION_SPACE_JOINT_DELTA,
)

Q_NOW = np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398])  # 'libero' reset pose
EE_POS_NOW = np.array([0.45, 0.0, 0.35])


def test_detects_joint_absolute_with_gripper():
    row = np.append(Q_NOW + 0.01, 0.5)
    chunk = np.tile(row, (10, 1))
    result = detect_action_space(chunk, Q_NOW, EE_POS_NOW)
    assert result.spec.kind == ACTION_SPACE_JOINT_ABSOLUTE
    assert result.spec.dim == 8
    assert result.spec.has_gripper


def test_detects_joint_delta_no_gripper():
    chunk = np.tile(np.full(7, 0.01), (10, 1))
    result = detect_action_space(chunk, Q_NOW, EE_POS_NOW)
    assert result.spec.kind == ACTION_SPACE_JOINT_DELTA
    assert not result.spec.has_gripper


def test_detects_ee_absolute_quat_with_gripper():
    row = np.concatenate([EE_POS_NOW + 0.01, [0.0, 0.0, 0.0, 1.0], [0.2]])
    chunk = np.tile(row, (10, 1))
    result = detect_action_space(chunk, Q_NOW, EE_POS_NOW)
    assert result.spec.kind == ACTION_SPACE_EE_ABSOLUTE
    assert result.spec.rotation_repr == "quat"
    assert result.spec.has_gripper


def test_detects_ee_delta_axis_angle_no_gripper():
    row = np.array([0.02, -0.01, 0.0, 0.05, 0.0, 0.0])
    chunk = np.tile(row, (10, 1))
    result = detect_action_space(chunk, Q_NOW, EE_POS_NOW)
    assert result.spec.kind == ACTION_SPACE_EE_DELTA
    assert result.spec.rotation_repr == "axis_angle"


def test_low_confidence_when_shape_matches_nothing():
    chunk = np.zeros((5, 13))  # no known action space has D=13
    result = detect_action_space(chunk, Q_NOW, EE_POS_NOW)
    assert result.spec.confidence == 0.0


def test_gripper_convention_pm1_detected():
    row = np.append(Q_NOW + 0.01, -1.0)
    chunk = np.tile(row, (10, 1))
    chunk[:, -1] = np.linspace(-1.0, 1.0, 10)
    result = detect_action_space(chunk, Q_NOW, EE_POS_NOW)
    assert result.spec.gripper_convention == "-1_1"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
