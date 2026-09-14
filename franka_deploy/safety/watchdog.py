"""Watches *measured* joint velocity for sustained oscillation.

The reference filter and EMA smoother both shape the commanded path, but
that's still only half the defense described in the project plan: both
assume the target is reasonable and the robot tracks it as intended.
Sustained vibration can still show up in what the robot actually measures
(gear backlash, an underdamped policy chasing its own noise at its
inference rate, resonance) without ever producing a single tick that trips
the jerk/accel clamp. This watches ``dq`` from robot state -- independent
of the command path -- and trips a hard stop if it looks like vibration
rather than directed motion.

Heuristic: velocity that keeps flipping sign every tick or two is
oscillation; smooth accel/decel motion has long same-sign runs. This is a
simple, interpretable proxy, not a spectral analysis -- the threshold needs
real-hardware tuning once there's an actual noisy policy to trip it against.

**2026-07-31 fix**: a joint that isn't moving reads tiny sensor noise
straddling zero, not a clean 0.0 -- ``np.sign()`` on that noise flips almost
every tick, which triggered a live false trip on a completely smooth
panda->fr3_ready move (only J2/J4/J6 were actually moving; J1/J3/J5/J7's
near-zero noise alone tripped the ``any(ratio > threshold)`` check). Sign
changes now only count between sample pairs that both clear
``velocity_deadzone`` -- a stationary joint's noise no longer looks like
oscillation just because it straddles zero.
"""

from __future__ import annotations

import collections
from typing import Deque

import numpy as np


class OscillationTripped(RuntimeError):
    """Raised when OscillationWatchdog decides the robot is vibrating."""


class OscillationWatchdog:
    def __init__(
        self,
        n_dims: int = 7,
        window_size: int = 50,
        sign_change_ratio_threshold: float = 0.5,
        trip_count_threshold: int = 3,
        velocity_deadzone: float = 0.02,
    ):
        self.n_dims = n_dims
        self.window_size = window_size
        self.threshold = sign_change_ratio_threshold
        self.trip_count_threshold = trip_count_threshold
        self.velocity_deadzone = velocity_deadzone
        self._history: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._consecutive_trips = 0

    def reset(self) -> None:
        self._history.clear()
        self._consecutive_trips = 0

    def update(self, dq: np.ndarray) -> bool:
        """Feed one tick's measured joint velocity. Returns True once
        sustained oscillation has been seen for ``trip_count_threshold``
        consecutive full windows (debounced, so one noisy window doesn't
        trip it)."""
        self._history.append(np.asarray(dq, dtype=float).copy())
        if len(self._history) < self.window_size:
            return False

        arr = np.array(self._history)  # (window_size, n_dims)
        signs = np.sign(arr)
        clearly_moving = np.abs(arr) > self.velocity_deadzone
        # A real sign change needs both samples clearly outside the noise
        # floor and opposite sign -- a stationary joint's noise (which
        # straddles zero without ever really moving) has few or no such
        # pairs, so it can't manufacture a high ratio on its own.
        both_moving = clearly_moving[:-1] & clearly_moving[1:]
        opposite_sign = (signs[:-1] * signs[1:]) < 0
        sign_changes = np.sum(both_moving & opposite_sign, axis=0)
        valid_pairs = np.sum(both_moving, axis=0)
        ratio = np.divide(
            sign_changes, valid_pairs, out=np.zeros_like(sign_changes, dtype=float), where=valid_pairs > 0
        )

        if np.any(ratio > self.threshold):
            self._consecutive_trips += 1
        else:
            self._consecutive_trips = 0
        return self._consecutive_trips >= self.trip_count_threshold
