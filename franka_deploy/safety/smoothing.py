"""Exponential smoothing on raw targets, applied before the reference filter.

The jerk-limited reference filter (robot/reference_filter.py) bounds the
*commanded* jerk/accel/velocity, but that guarantee assumes its ``target``
input is itself something reasonable to chase -- it doesn't know or care
where the target came from. A policy's raw per-tick output can still be
noisier than a human teleop hand (see the project plan's two-layer defense
decision); this softens that noise before it ever reaches the filter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class EMASmoother:
    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._value: Optional[np.ndarray] = None

    def reset(self, q0: np.ndarray) -> None:
        self._value = np.asarray(q0, dtype=float).copy()

    def step(self, raw_target: np.ndarray) -> np.ndarray:
        raw_target = np.asarray(raw_target, dtype=float)
        if self._value is None:
            self._value = raw_target.copy()
        else:
            self._value = self.alpha * raw_target + (1.0 - self.alpha) * self._value
        return self._value.copy()
