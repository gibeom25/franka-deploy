"""Synthetic camera for dry-run / no-hardware testing -- lets the dashboard's
detect/schema-editor flow be exercised on a dev machine or before cameras
are physically connected, same purpose as fr3_policy_client.py's
``--dry-run`` synthetic image."""

from __future__ import annotations

import numpy as np


class MockCamera:
    def __init__(self, width: int = 640, height: int = 480, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._shape = (height, width, 3)

    def read(self):
        rgb = self._rng.integers(0, 255, self._shape, dtype=np.uint8)
        depth = np.zeros((self._shape[0], self._shape[1], 1), dtype=np.uint16)
        return rgb, depth

    def stop(self) -> None:
        pass
