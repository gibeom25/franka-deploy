"""Background-threaded camera reader: one dedicated thread per camera,
continuously grabbing frames so callers (the policy loop, a dashboard
preview endpoint) get the latest frame instantly instead of blocking on
`camera.read()` (a RealSense `wait_for_frames()` call).

Two reasons this is its own thread per camera, not read inline:
1. Isolation -- after the live GIL-contention incidents that motivated
   the robot-node process split (see control_loop.py's module docstring),
   camera I/O is kept off the policy tick path too, on general principle:
   nothing blocking shares a thread with time-sensitive work if it doesn't
   have to.
2. A dashboard camera-check needs a live frame independent of whether a
   control session is even running -- a shared background reader, not
   something only the policy loop owns, is what makes that possible.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from franka_deploy.cameras import CameraConfig, build_cameras


class _CameraWorker:
    def __init__(self, camera):
        self._camera = camera
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._error: Optional[str] = None
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                rgb, _depth = self._camera.read()
                with self._lock:
                    self._latest = rgb
                    self._error = None
            except Exception as e:  # noqa: BLE001 -- keep the worker alive, surface the error instead
                with self._lock:
                    self._error = str(e)

    def get_latest(self):
        with self._lock:
            return self._latest, self._error

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        try:
            self._camera.stop()
        except Exception:  # noqa: BLE001
            pass


class CameraManager:
    """Owns every configured camera's background reader. One instance
    shared by the dashboard preview endpoints and ControlLoop, so both see
    the same live frames and a camera is never opened twice."""

    def __init__(self):
        self._workers: dict[str, _CameraWorker] = {}

    def start(self, configs: list[CameraConfig], ready_timeout: float = 5.0) -> None:
        """Stops whatever is currently running FIRST, then builds fresh
        camera objects from configs -- in that order. Building the new
        RealSenseCamera (which opens a pipeline in __init__) before the old
        one released the same physical device was a real, observed bug:
        the two pipelines raced for the same USB device and start() lost
        that race often enough to matter. Taking configs (not pre-built
        camera objects) instead of leaving construction to the caller is
        what lets this method guarantee the order.

        Blocks until every camera has produced its first frame (or errored),
        up to ready_timeout -- without this, a caller that reads a frame
        immediately after start() (e.g. detect()'s first observation) can
        race the background thread's very first capture and see "no frame
        yet" even though the camera is working fine."""
        self.stop()
        time.sleep(0.3)  # let the just-released USB pipeline fully settle before reopening
        cameras = build_cameras(configs)
        self._workers = {role: _CameraWorker(cam) for role, cam in cameras.items()}
        deadline = time.monotonic() + ready_timeout
        for w in self._workers.values():
            while time.monotonic() < deadline:
                latest, error = w.get_latest()
                if latest is not None or error is not None:
                    break
                time.sleep(0.01)

    def roles(self) -> list[str]:
        return list(self._workers)

    def get_latest(self, role: str):
        w = self._workers.get(role)
        if w is None:
            return None, f"camera role {role!r} not started"
        return w.get_latest()

    def stop(self) -> None:
        for w in self._workers.values():
            w.stop()
        self._workers = {}
