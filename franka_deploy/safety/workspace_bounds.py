"""Cartesian workspace fence, defined by teleoperating (hand-guiding) the
robot to its safe-region corners rather than guessed/hardcoded -- "너무
말도 안되는 움직임" (nonsensical motion) observed live is exactly what
joint-space limits and the reference filter don't catch: a target can be
smooth, within joint limits, and still walk the EE somewhere the operator
never wanted it to go (e.g. a bad EE_absolute/EE_delta interpretation).
This is an independent, dumb-simple check on top of everything else.

Recording: connect read_only, put the robot in Programming (white) mode
on Desk (see [[franka-hand-guiding]]), hand-guide it to each boundary
point, click "add point" per point. No fewer than 2 points -> bounds are
undefined and nothing is enforced (fail open on missing configuration,
fail closed once points exist -- see `check()`).

Deliberately an axis-aligned bounding box, not a convex hull or anything
fancier: for a safety check, "obviously correct to read" beats "tighter
fit." A margin shrinks the box inward for extra headroom.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class WorkspaceExceeded(RuntimeError):
    """Raised when a checked position falls outside the recorded box."""


@dataclass
class WorkspaceBounds:
    points: list = field(default_factory=list)  # list[[x,y,z]], meters, base frame
    margin: float = 0.02  # meters, shrinks the box inward

    def add_point(self, pos) -> None:
        self.points.append([float(v) for v in pos])

    def remove_point(self, index: int) -> None:
        del self.points[index]

    def clear(self) -> None:
        self.points = []

    @property
    def enabled(self) -> bool:
        return len(self.points) >= 2

    def bbox(self):
        """(lo, hi), each a 3-vector -- or None if fewer than 2 points."""
        if not self.enabled:
            return None
        arr = np.asarray(self.points, dtype=float)
        lo = arr.min(axis=0) + self.margin
        hi = arr.max(axis=0) - self.margin
        return lo, hi

    def check(self, pos) -> None:
        """No-op if bounds aren't defined yet (fewer than 2 points) --
        undefined bounds means nothing to enforce, not "anything goes
        forever": the dashboard always shows point count so an operator
        can tell an empty fence from an intentional one."""
        bbox = self.bbox()
        if bbox is None:
            return
        lo, hi = bbox
        pos = np.asarray(pos, dtype=float)
        if np.any(pos < lo) or np.any(pos > hi):
            raise WorkspaceExceeded(
                f"EE position {np.round(pos, 3).tolist()} outside recorded workspace "
                f"bounds lo={np.round(lo, 3).tolist()} hi={np.round(hi, 3).tolist()}"
            )

    def to_dict(self) -> dict:
        bbox = self.bbox()
        return {
            "points": self.points,
            "margin": self.margin,
            "enabled": self.enabled,
            "lo": bbox[0].tolist() if bbox else None,
            "hi": bbox[1].tolist() if bbox else None,
        }
