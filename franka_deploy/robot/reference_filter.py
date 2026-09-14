"""Jerk-limited reference filter for tracking a live setpoint.

Ported from the control math in ``~/teleop-franka/gello_software/gello/robots/
franka_fr3.py`` (``_control_loop``, critically-damped second-order filter).
That file's docstring explains why this exists: pylibfranka's active-control
API (``start_joint_position_control`` -> ``readOnce``/``writeOnce``) sends
whatever command you give it straight to the robot with zero shaping -- unlike
libfranka's blocking ``Robot::control()`` path, it does not run
``limitRate()``/``lowpassFilter()`` itself. Without this filter, a noisy or
discontinuous target (a human hand on gello_software's leader, or here a
policy's raw action) produces a jerk spike that trips
``joint_motion_generator_acceleration_discontinuity`` at best, and drives
sustained vibration into the joints at worst.

This is deliberately just the math, with no pylibfranka dependency, so it can
be unit-tested without hardware (see ``tests/test_reference_filter.py``).
"""

from __future__ import annotations

import numpy as np


class JerkLimitedReferenceFilter:
    """Critically-damped 2nd-order filter, saturated in jerk/accel/velocity.

    Call :meth:`reset` once with the robot's actual starting position, then
    call :meth:`step` once per control tick with the current desired target.
    The returned command changes smoothly even if ``target`` jumps or is
    noisy -- the filter is the safety boundary, not the caller.

    Args:
        n_dims: dimensionality (7 for joints, 6 for a Cartesian twist, ...).
        dt: control period in seconds (1e-3 for the FCI's 1 kHz loop).
        v_max, a_max, j_max: velocity / acceleration / jerk saturation
            limits. Scalar (applied to all dims) or per-dim array. Defaults
            match the values validated live on this FR3
            (max_joint_velocity=1.0 rad/s, max_joint_acceleration=4.0 rad/s^2,
            max_joint_jerk=3000, comfortably under libfranka's
            ``kMaxJointJerk`` of 5000) -- don't raise these without
            re-validating on hardware first.
        wn: natural frequency of the reference filter (rad/s). Higher tracks
            the target more aggressively; lower is smoother. kp = wn^2,
            kd = 2*wn (critical damping -> no overshoot by construction).
    """

    def __init__(
        self,
        n_dims: int,
        dt: float = 1e-3,
        v_max: float | np.ndarray = 1.0,
        a_max: float | np.ndarray = 4.0,
        j_max: float | np.ndarray = 3000.0,
        wn: float = 10.0,
    ):
        self.n_dims = n_dims
        self.dt = float(dt)
        self.v_max = np.broadcast_to(np.asarray(v_max, dtype=float), (n_dims,)).copy()
        self.a_max = np.broadcast_to(np.asarray(a_max, dtype=float), (n_dims,)).copy()
        self.j_max = np.broadcast_to(np.asarray(j_max, dtype=float), (n_dims,)).copy()
        self.kp = float(wn) ** 2
        self.kd = 2.0 * float(wn)

        self.q_cmd = np.zeros(n_dims)
        self.qd_cmd = np.zeros(n_dims)
        self._acc_prev = np.zeros(n_dims)

    def reset(self, q0: np.ndarray) -> None:
        """Latch the filter to an actual starting position (no ramp-in jump)."""
        self.q_cmd = np.asarray(q0, dtype=float).copy()
        self.qd_cmd = np.zeros(self.n_dims)
        self._acc_prev = np.zeros(self.n_dims)

    def step(self, target: np.ndarray) -> np.ndarray:
        """Advance the filter one tick toward ``target``. Returns the command."""
        target = np.asarray(target, dtype=float)
        err = target - self.q_cmd
        acc_target = np.clip(
            self.kp * err - self.kd * self.qd_cmd, -self.a_max, self.a_max
        )
        dacc_max = self.j_max * self.dt
        acc = np.clip(acc_target, self._acc_prev - dacc_max, self._acc_prev + dacc_max)

        qd_new = np.clip(self.qd_cmd + acc * self.dt, -self.v_max, self.v_max)
        # Feed back the acceleration that actually survived the velocity
        # clamp (not the pre-clamp value) so acc_prev doesn't drift away from
        # reality and spike the moment the clamp releases.
        self._acc_prev = (qd_new - self.qd_cmd) / self.dt
        self.qd_cmd = qd_new
        self.q_cmd = self.q_cmd + self.qd_cmd * self.dt
        return self.q_cmd.copy()
