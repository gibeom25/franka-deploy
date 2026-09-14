"""Safety constants ported from the values validated live on this FR3.

Source: ``~/teleop-franka/gello_software/gello/robots/franka_fr3.py``. These
are not re-derived from the datasheet -- they were tuned against what this
specific robot's collision reflex actually does (see that file's comments for
the reasoning), so keep them in sync with that file if it changes rather than
re-deriving independently.
"""

from __future__ import annotations

import numpy as np

# Franka Hand stroke (m).
MAX_GRIPPER_WIDTH = 0.08

# Collision-reflex torque thresholds (N*m), J1..J7. NOT the datasheet
# actuation limits (those trip on ordinary contact -- see franka_fr3.py's
# FR3_MAX_JOINT_TORQUE comment). Sized from the 100 N cartesian force reflex,
# which fires first in any sustained contact.
FR3_COLLISION_TORQUE = [100.0, 100.0, 100.0, 100.0, 40.0, 40.0, 40.0]
FR3_COLLISION_FORCE = 100.0  # N, cartesian

DEFAULT_JOINT_IMPEDANCE = [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]

# Reference-filter saturation defaults (see reference_filter.py). Comfortably
# below libfranka's hard per-joint limits (2.62 rad/s, 10.0 rad/s^2) and
# kMaxJointJerk (5000). Validated live via gello teleop; raise only after
# re-validating on hardware, not by assumption -- a control loop whose dt
# drifts from the nominal 1 ms (e.g. a heavy policy inference call blocking
# the same process) makes any headroom here shrink in practice.
DEFAULT_V_MAX = 1.0
DEFAULT_A_MAX = 4.0
DEFAULT_J_MAX = 3000.0
DEFAULT_FILTER_WN = 10.0

# FR3 joint position limits (rad), J1..J7 -- same values as franka_fr3.py.
# Soft safety floor; the runtime should clip commanded targets well inside
# these, not rely on them as the only guard.
FR3_Q_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_Q_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Named reset poses (rad), mirroring franka_fr3.py's FR3_RESET_POSES. All four
# are inside FR3_Q_LOWER/FR3_Q_UPPER.
FR3_RESET_POSES = {
    "fr3_ready": np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]),
    "panda": np.array([0.0, 0.0, 0.0, -1.570796, 0.0, 1.570796, 0.785398]),
    "libero": np.array(
        [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398]
    ),
}
DEFAULT_RESET_POSE = "panda"

# Margin (rad) kept inside FR3_Q_LOWER/UPPER by clip_to_joint_limits. A
# policy target landing exactly on the hard limit still leaves the reference
# filter's own overshoot-free tracking no room at all; back off a bit so the
# clip isn't itself the thing that trips a limit reflex.
DEFAULT_JOINT_LIMIT_MARGIN = 0.05


def clip_to_joint_limits(q: np.ndarray, margin: float = DEFAULT_JOINT_LIMIT_MARGIN) -> np.ndarray:
    """Clamp a commanded joint target well inside the soft position limits.

    A last-resort floor, not a substitute for a policy staying in-distribution
    -- silently clipping a wildly wrong target still lets the robot move
    toward a nonsensical pose, just not past the joint limit.
    """
    return np.clip(np.asarray(q, dtype=float), FR3_Q_LOWER + margin, FR3_Q_UPPER - margin)
