"""Rotation-representation conversions not already in the vendored
fr3_kinematics.py (which only has axis-angle <-> matrix, since that's the
only representation manipulation-stack's EE-delta checkpoints use). Added
here, not edited into the vendored file, so it stays a byte-identical copy
of the validated original.

Needed because a policy server's response format is user-defined and may
express EE rotation as a quaternion or a 6D rotation (Zhou et al. continuous
representation) instead of axis-angle.
"""

from __future__ import annotations

import numpy as np


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] (scalar-last, the common robotics/ROS/scipy convention) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def rot6d_to_rot(six: np.ndarray) -> np.ndarray:
    """Zhou et al. 2019 continuous 6D rotation -> 3x3, via Gram-Schmidt.
    Inverse of fr3_kinematics.py's ``_rot6d`` (which goes matrix -> 6D)."""
    six = np.asarray(six, dtype=np.float64)
    a1, a2 = six[:3], six[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / max(np.linalg.norm(a2_orth), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)
