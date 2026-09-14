"""FR3 robot runtime: connect, configure safety, and track a live setpoint.

Architecture (see the project plan / README for the full rationale): a 1 kHz
background thread continuously advances a command toward whatever the latest
target is, through :class:`~policy_runner.robot.reference_filter.
JerkLimitedReferenceFilter`. Callers -- whether a blocking ``movej()``, a
teleop stream, or a policy's ``predict()`` loop -- only ever update the
target; they never touch the 1 kHz loop directly. This mirrors
``gello_software/gello/robots/franka_fr3.py``, which validated this pattern
live on this exact robot.

Safety is layered, not implicit:
  1. ``set_collision_behavior`` / ``set_joint_impedance`` at connect time.
  2. The jerk/accel/velocity-saturated reference filter every tick.
  3. ``read_only=True`` connects and streams state but commands no motion at
     all -- always validate a new policy checkpoint's predicted actions this
     way before arming real motion.
Further layers (stale-action watchdog, oscillation detection on measured
``dq``, gripper debounce) live in ``policy_runner.safety`` and wrap this
runtime rather than living inside it.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from franka_deploy.robot.reference_filter import JerkLimitedReferenceFilter
from franka_deploy.safety.limits import (
    DEFAULT_A_MAX,
    DEFAULT_FILTER_WN,
    DEFAULT_J_MAX,
    DEFAULT_V_MAX,
    DEFAULT_JOINT_IMPEDANCE,
    FR3_COLLISION_FORCE,
    FR3_COLLISION_TORQUE,
)


class FR3Runtime:
    """Joint-space runtime for the FR3, driven through ``pylibfranka``.

    Args:
        robot_ip: FCI address.
        read_only: connect and stream state but never command motion. Use
            this to validate a new policy's predicted actions before letting
            it touch the robot.
        enforce_rt: ``RealtimeConfig.kEnforce`` (True) or ``kIgnore``.
        v_max/a_max/j_max/filter_wn: reference-filter saturation. Defaults
            are the values validated live via gello teleop -- see
            ``policy_runner.safety.limits``.
        collision_torque/collision_force: collision-reflex thresholds. None
            uses the values tuned on this robot (``safety.limits``).
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        read_only: bool = True,
        enforce_rt: bool = True,
        v_max: float = DEFAULT_V_MAX,
        a_max: float = DEFAULT_A_MAX,
        j_max: float = DEFAULT_J_MAX,
        filter_wn: float = DEFAULT_FILTER_WN,
        collision_torque: Optional[list] = None,
        collision_force: float = FR3_COLLISION_FORCE,
    ):
        import pylibfranka as pf

        self._pf = pf
        self._read_only = read_only
        self._dt = 1e-3

        rt = pf.RealtimeConfig.kEnforce if enforce_rt else pf.RealtimeConfig.kIgnore
        print(f"[FR3Runtime] connecting to {robot_ip} (realtime={'enforce' if enforce_rt else 'ignore'}, "
              f"read_only={read_only})")
        self.robot = pf.Robot(robot_ip, rt)

        # Clear a leftover reflex/error state so a plain restart works
        # without touching Desk. Harmless if already Idle.
        try:
            self.robot.automatic_error_recovery()
        except Exception as e:  # noqa: BLE001
            print(f"[FR3Runtime] automatic error recovery failed: {e}")

        torque_thresh = list(collision_torque or FR3_COLLISION_TORQUE)
        self.robot.set_collision_behavior(
            torque_thresh, torque_thresh,
            [collision_force] * 6, [collision_force] * 6,
        )
        self.robot.set_joint_impedance(DEFAULT_JOINT_IMPEDANCE)

        st = self.robot.read_once()
        q0 = np.asarray(st.q, dtype=float)
        print(f"[FR3Runtime] connected. q = {np.round(q0, 3)}  mode = {st.robot_mode}")

        self._filter = JerkLimitedReferenceFilter(
            n_dims=7, dt=self._dt, v_max=v_max, a_max=a_max, j_max=j_max, wn=filter_wn
        )

        self._lock = threading.Lock()
        self._q = q0.copy()
        self._dq = np.zeros(7)
        self._target_q = q0.copy()
        self._ee_pose = np.asarray(st.O_T_EE, dtype=float)
        self._success_rate = 1.0
        self._control_error: Optional[str] = None
        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._read_only_loop if read_only else self._control_loop,
            daemon=True,
        )
        self._thread.start()
        if not read_only:
            time.sleep(0.2)
            if self._control_error is not None:
                raise RuntimeError(f"[FR3Runtime] control loop failed to start: {self._control_error}")

    # --------------------------------------------------------------- public
    @property
    def read_only(self) -> bool:
        return self._read_only

    def get_state(self) -> dict:
        if self._control_error is not None:
            raise RuntimeError(f"[FR3Runtime] control loop is dead: {self._control_error}")
        with self._lock:
            return {
                "q": self._q.copy(),
                "dq": self._dq.copy(),
                "ee_pose": self._ee_pose.copy(),
                "control_command_success_rate": self._success_rate,
            }

    def set_joint_target(self, q: np.ndarray) -> None:
        """Non-blocking: update the setpoint the background loop tracks."""
        if self._read_only:
            raise RuntimeError("[FR3Runtime] read_only=True: motion is disabled")
        with self._lock:
            self._target_q = np.asarray(q, dtype=float).copy()

    def movej(self, q_target: np.ndarray, timeout: float = 15.0, tol: float = 0.01) -> bool:
        """Blocking joint move: set the target and wait until reached.

        A thin wrapper over the continuously-tracking loop, not a separate
        trajectory type -- see the project plan's control-architecture
        decision. Returns False on timeout (does not raise, so a caller can
        decide whether that's fatal).
        """
        self.set_joint_target(q_target)
        q_target = np.asarray(q_target, dtype=float)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if np.abs(self.get_state()["q"] - q_target).max() < tol:
                return True
            time.sleep(0.02)
        return False

    @property
    def control_command_success_rate(self) -> float:
        with self._lock:
            return self._success_rate

    def stop(self, decel_timeout: float = 3.0, decel_vel_tol: float = 0.01) -> None:
        """Stop the control loop.

        Declaring ``motion_finished`` while the robot is still moving at any
        real velocity aborts with ``joint_motion_generator_acceleration_
        discontinuity`` ("Motion finished commanded, but the robot is still
        moving!") -- observed live when an OscillationWatchdog trip called
        this mid-motion. So for real motion, freeze the target at the
        current measured position first and let the reference filter's own
        jerk/accel/velocity limits decelerate it to a stop (typically well
        under a second at the default limits), polling *measured* dq, before
        signalling the loop to end.
        """
        if not self._read_only and self._control_error is None:
            with self._lock:
                self._target_q = self._q.copy()
            deadline = time.time() + decel_timeout
            while time.time() < deadline:
                if self._control_error is not None:
                    break
                with self._lock:
                    dq = self._dq.copy()
                if np.abs(dq).max() < decel_vel_tol:
                    break
                time.sleep(0.01)

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.robot.stop()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ loops
    def _read_only_loop(self) -> None:
        while not self._stop.is_set():
            try:
                st = self.robot.read_once()
                with self._lock:
                    self._q = np.asarray(st.q, dtype=float)
                    self._dq = np.asarray(st.dq, dtype=float)
                    self._ee_pose = np.asarray(st.O_T_EE, dtype=float)
                    self._success_rate = float(st.control_command_success_rate)
            except Exception as e:  # noqa: BLE001
                self._control_error = str(e)
                print(f"[FR3Runtime] read-only loop error: {e}")
                break
            time.sleep(0.001)

    def _control_loop(self) -> None:
        pf = self._pf
        try:
            ctrl = self.robot.start_joint_position_control(pf.ControllerMode.JointImpedance)

            # Latch to the robot's own q_d, not measured q -- the robot-side
            # motion generator differentiates the incoming stream starting
            # from q_d, so any q-vs-q_d gap reads as a phantom velocity step.
            # Same convention as franka_fr3.py / libfranka's own examples.
            state, _ = ctrl.readOnce()
            with self._lock:
                self._q = np.asarray(state.q, dtype=float)
                q_d = np.asarray(state.q_d, dtype=float)
                gap = float(np.abs(q_d - self._q).max())
                if gap > 0.05:
                    raise RuntimeError(
                        f"stale desired pose: max|q - q_d| = {gap:.3f} rad; "
                        "run error recovery or re-open the brakes, then relaunch"
                    )
                self._target_q = q_d.copy()
            self._filter.reset(q_d)
            cmd = pf.JointPositions(list(self._filter.q_cmd))
            ctrl.writeOnce(cmd)

            while not self._stop.is_set():
                state, _ = ctrl.readOnce()
                with self._lock:
                    target = self._target_q.copy()
                    self._q = np.asarray(state.q, dtype=float)
                    self._dq = np.asarray(state.dq, dtype=float)
                    self._ee_pose = np.asarray(state.O_T_EE, dtype=float)
                    self._success_rate = float(state.control_command_success_rate)

                q_cmd = self._filter.step(target)

                cmd = pf.JointPositions(list(q_cmd))
                if self._stop.is_set():
                    cmd.motion_finished = True
                ctrl.writeOnce(cmd)
        except Exception as e:  # noqa: BLE001
            self._control_error = str(e)
            print(f"[FR3Runtime] CONTROL LOOP ABORTED: {e}")
