"""Binary grasp/release gripper control -- discretized, debounced.

The Franka Hand can't servo continuous width from a policy's raw output:
``move()`` unloads and backs off the moment it meets resistance short of the
target width (no held force), and the hand physically takes ~0.8s per
stroke. ``gello_software/gello/robots/franka_fr3.py``'s ``_gripper_loop``
already solved this for teleop with a binary grasp/release state machine
with hysteresis; ``GripperHysteresis`` below is that same decision logic
pulled out as pure, hardware-free code so it's independently testable, plus
an explicit minimum dwell time (``min_dwell_s``) beyond what teleop needed --
a policy can chatter across the hysteresis band faster than a human hand,
and every grasp/release cycle is real mechanical wear on the actuator.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import numpy as np

from franka_deploy.safety.limits import MAX_GRIPPER_WIDTH


class GripperHysteresis:
    """Pure decision logic -- no I/O, fully unit-testable.

    ``trigger`` is normalized 0=open .. 1=closed (gello convention).
    :meth:`decide` returns ``"grasp"``, ``"move"`` (open), or ``None``
    (hold current state -- includes being inside the hysteresis band or
    still within ``min_dwell_s`` of the last transition).
    """

    def __init__(self, close_at: float = 0.6, open_at: float = 0.2, min_dwell_s: float = 0.5):
        if not open_at < close_at:
            raise ValueError(f"open_at ({open_at}) must be < close_at ({close_at})")
        self.close_at = close_at
        self.open_at = open_at
        self.min_dwell_s = min_dwell_s
        self._closed = False
        self._last_change_t = -math.inf  # first crossing is never debounced

    @property
    def closed(self) -> bool:
        return self._closed

    def decide(self, trigger: float, now: float) -> Optional[str]:
        if now - self._last_change_t < self.min_dwell_s:
            return None
        if not self._closed and trigger >= self.close_at:
            self._closed = True
            self._last_change_t = now
            return "grasp"
        if self._closed and trigger <= self.open_at:
            self._closed = False
            self._last_change_t = now
            return "move"
        return None


class GripperRuntime:
    """Owns the real pylibfranka ``Gripper`` + a background decision thread.

    Mirrors ``franka_fr3.py``'s ``_gripper_loop`` (same blocking-call shape,
    same edge-triggered commands) but delegates the actual open/close
    decision to :class:`GripperHysteresis` so that logic isn't only
    exercised live on hardware.
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        close_at: float = 0.6,
        open_at: float = 0.2,
        min_dwell_s: float = 0.5,
        speed: float = 0.1,
        grasp_force: float = 40.0,
        epsilon: float = 0.08,  # must span the full stroke -- see franka_fr3.py's _gripper_loop docstring
        home_on_connect: bool = False,
        poll_period_s: float = 0.05,
    ):
        import pylibfranka as pf

        self._gripper = pf.Gripper(robot_ip)
        if home_on_connect:
            print("[GripperRuntime] homing gripper (this moves the fingers)...")
            self._gripper.homing()

        self._hysteresis = GripperHysteresis(close_at, open_at, min_dwell_s)
        self._speed = speed
        self._grasp_force = grasp_force
        self._epsilon = epsilon
        self._poll_period_s = poll_period_s

        gs = self._gripper.read_once()
        self._lock = threading.Lock()
        self._width = float(gs.width)
        self._target_trigger = 1.0 - self._width / MAX_GRIPPER_WIDTH
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_target(self, trigger: float) -> None:
        with self._lock:
            self._target_trigger = float(np.clip(trigger, 0.0, 1.0))

    def get_width(self) -> float:
        with self._lock:
            return self._width

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._gripper.stop()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                trigger = self._target_trigger
            decision = self._hysteresis.decide(trigger, time.time())
            try:
                if decision == "grasp":
                    ok = self._gripper.grasp(
                        0.0, self._speed, self._grasp_force,
                        epsilon_inner=self._epsilon, epsilon_outer=self._epsilon,
                    )
                    if not ok:
                        print("[GripperRuntime] grasp reported failure")
                elif decision == "move":
                    self._gripper.move(MAX_GRIPPER_WIDTH, self._speed)
                gs = self._gripper.read_once()
                with self._lock:
                    self._width = float(gs.width)
            except Exception as e:  # noqa: BLE001
                print(f"[GripperRuntime] {decision or 'poll'} failed: {e}")
            time.sleep(self._poll_period_s)
