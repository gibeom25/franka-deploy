"""Two-rate control orchestrator: FR3Runtime's 1 kHz local tracking loop
(running in a SEPARATE process, robot/robot_node.py, talked to over ZMQ via
robot/zmq_client.py -- see that module's docstring for why: real hardware
faults were observed when IK/web/HTTP work shared a process/GIL with the
control thread) + a background policy-inference loop in THIS process that
talks to an arbitrary remote server and feeds it new setpoints.

The chunk-boundary-stall fix (fire the next /predict CHUNK_LEAD ticks
before the current chunk runs out; drop however many leading indices have
actually gone stale by wall-clock time) is ported from
``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py`` --
see [[franka-async-chunking]] in the project memory for why this specific
scheme (skip = actual elapsed ticks, not a hardcoded K) matters.

State machine mirrors franka-policy-runner's staged-rollout culture
(preflight -> read_only dry run -> confirm -> real motion):

    IDLE -> CONNECTED (FR3Runtime open, read_only)
          -> DETECTING (one reset+predict cycle, response_detect.py runs)
          -> AWAITING_CONFIRM (spec shown to the dashboard, no motion yet)
          -> ARMED (spec confirmed by the user)
          -> RUNNING -> STOPPED / ERROR
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from franka_deploy.robot.zmq_client import ZMQRobotClient
from franka_deploy.safety.limits import clip_to_joint_limits
from franka_deploy.safety.smoothing import EMASmoother
from franka_deploy.safety.watchdog import OscillationTripped, OscillationWatchdog
from franka_deploy.schema.apply import ActionSpaceAdapter
from franka_deploy.schema.client import PolicyClient
from franka_deploy.schema.response_detect import detect_action_space
from franka_deploy.schema.sources import SourceContext, robot_state_dict
from franka_deploy.schema.spec import ActionSpaceSpec, DetectionResult, RequestSpec


class State(str, enum.Enum):
    IDLE = "idle"
    CONNECTED = "connected"
    DETECTING = "detecting"
    AWAITING_CONFIRM = "awaiting_confirm"
    ARMED = "armed"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LoopConfig:
    fps: float = 20.0
    exec_horizon: int = 10
    lead_ticks: int = 2
    max_step_rad: float = 0.5   # per-tick |target-measured| clamp -- independent of, and in
                                 # addition to, the reference filter's own v/a/j limits
    ema_alpha: float = 0.3
    watchdog_enabled: bool = True


@dataclass
class Telemetry:
    state: State = State.IDLE
    q: Optional[list] = None
    dq: Optional[list] = None
    ee_pos: Optional[list] = None
    control_command_success_rate: float = 0.0
    last_predict_ms: float = 0.0
    n_replans: int = 0
    n_late: int = 0
    error: Optional[str] = None


def ee_pose_to_matrix(flat_o_t_ee: np.ndarray) -> np.ndarray:
    """libfranka's O_T_EE is a flat column-major 4x4 (std::array<double,16>).
    FR3Runtime.get_state() passes it through unreshaped -- reshape here
    rather than in the hot control loop's every caller."""
    return np.asarray(flat_o_t_ee, dtype=float).reshape(4, 4, order="F")


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> [x, y, z, w]. Small and local; not worth a dependency."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


class ControlLoop:
    """Owns the robot runtime, camera reads, policy client, and the
    background policy thread. One instance per running session."""

    def __init__(self, robot_node_address: str, camera_manager, request_spec: RequestSpec,
                 loop_cfg: Optional[LoopConfig] = None, safety_kwargs: Optional[dict] = None,
                 gripper_runtime=None):
        self._robot_node_address = robot_node_address
        self._camera_manager = camera_manager  # cameras.manager.CameraManager, own background threads
        self._client = PolicyClient(request_spec)
        self._cfg = loop_cfg or LoopConfig()
        self._safety_kwargs = safety_kwargs or {}
        self._gripper_runtime = gripper_runtime  # optional GripperRuntime, set_target(0..1)

        self._runtime: Optional[ZMQRobotClient] = None
        self._smoother = EMASmoother(alpha=self._cfg.ema_alpha)
        self._watchdog = OscillationWatchdog()
        self._adapter: Optional[ActionSpaceAdapter] = None
        self.last_detection: Optional[DetectionResult] = None

        self._state = State.IDLE
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._telemetry = Telemetry()

    # ------------------------------------------------------------ lifecycle
    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _set_state(self, s: State) -> None:
        with self._lock:
            self._state = s

    def connect(self, read_only: bool = True) -> None:
        if self._runtime is None:
            self._runtime = ZMQRobotClient(self._robot_node_address)
        self._runtime.connect(read_only=read_only, **self._safety_kwargs)
        self._set_state(State.CONNECTED)

    def _read_cameras(self) -> dict:
        """Non-blocking: pulls whatever the CameraManager's background
        threads have already captured, rather than waiting on the camera
        here (see cameras/manager.py's docstring for why)."""
        out = {}
        for role in self._camera_manager.roles():
            rgb, error = self._camera_manager.get_latest(role)
            if rgb is not None:
                out[role] = rgb
        return out

    def _current_ctx_and_state(self):
        st = self._runtime.get_state()
        ee_mat = ee_pose_to_matrix(st["ee_pose"])
        ee_pos, ee_quat = ee_mat[:3, 3], _mat_to_quat(ee_mat[:3, :3])
        gripper = self._gripper_runtime.get_width() / 0.08 if self._gripper_runtime else 0.0
        robot_state = robot_state_dict(st["q"], st["dq"], ee_pos, ee_quat, gripper)
        ctx = SourceContext(cameras=self._read_cameras(), robot_state=robot_state)
        return ctx, st, ee_pos

    # -------------------------------------------------------------- staged rollout
    def detect(self, instruction: Optional[str] = None) -> DetectionResult:
        """One reset + predict cycle to classify the response format. Safe
        to call in read_only mode -- no motion is ever commanded here."""
        self._set_state(State.DETECTING)
        try:
            self._client.reset(instruction)
            ctx, st, ee_pos = self._current_ctx_and_state()
            chunk = self._client.predict(ctx)
            result = detect_action_space(chunk, st["q"], ee_pos)
        except Exception:
            # A bad server address/timeout must not strand the session in
            # DETECTING forever with no way to retry short of a full
            # disconnect -- fall back to CONNECTED, which /detect accepts.
            self._set_state(State.CONNECTED)
            raise
        self.last_detection = result
        self._set_state(State.AWAITING_CONFIRM)
        return result

    def confirm(self, spec: ActionSpaceSpec) -> None:
        """User has reviewed the detected (or manually chosen) spec. Arms
        the adapter; still commands no motion -- call start() for that."""
        self._adapter = ActionSpaceAdapter(spec)
        self._set_state(State.ARMED)

    def start(self, instruction: Optional[str] = None) -> None:
        """Ordered to keep GIL-contending Python work (HTTP calls, JSON)
        OFF the moment the 1 kHz control thread spins up -- a real
        `communication_constraints_violation` reflex was observed live when
        connect() + reset() + the first predict() all landed back-to-back
        with the control thread's startup. Network calls happen first
        (still read_only, nothing time-critical yet); the live connect
        happens last, immediately followed by a settle pause before any
        other Python work resumes on this thread."""
        if self._adapter is None:
            raise RuntimeError("call detect() then confirm() before start()")
        self._client.reset(instruction)
        if self._runtime is None or self._runtime.read_only:
            self.connect(read_only=False)
        time.sleep(0.2)  # let the new 1 kHz thread get its RT priority and settle
        q0 = self._runtime.get_state()["q"]
        self._smoother.reset(q0)
        self._watchdog.reset()
        self._stop_evt.clear()
        self._telemetry = Telemetry(state=State.RUNNING)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._set_state(State.RUNNING)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.state != State.ERROR:
            self._set_state(State.STOPPED)

    def estop(self) -> None:
        self.stop()
        if self._runtime is not None:
            self._runtime.stop()

    def get_telemetry(self) -> Telemetry:
        with self._lock:
            return self._telemetry

    # ------------------------------------------------------------- the loop
    def _run(self) -> None:
        cfg = self._cfg
        dt = 1.0 / cfg.fps
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-predict")
        pending = None
        t_obs = 0.0
        n_replans = n_late = 0
        try:
            ctx, _, _ = self._current_ctx_and_state()
            chunk = self._client.predict(ctx)
            n_replans = 1
            idx = 0
            t_next = time.monotonic()

            while not self._stop_evt.is_set():
                horizon = min(len(chunk), cfg.exec_horizon)

                # (1) chunk exhausted -> swap in the next one, dropping
                # however many indices have actually gone stale by wall
                # clock (nominally == lead_ticks, more if inference ran long).
                if idx >= horizon:
                    if pending is None:
                        t_obs = time.monotonic()
                        ctx, _, _ = self._current_ctx_and_state()
                        chunk = self._client.predict(ctx)
                    else:
                        if not pending.done():
                            n_late += 1
                        chunk = pending.result()
                        pending = None
                    n_replans += 1
                    horizon = min(len(chunk), cfg.exec_horizon)
                    skip = round((time.monotonic() - t_obs) / dt)
                    idx = min(max(skip, 0), horizon - 1)

                # (2) lead_ticks before the boundary -> fire the next
                # inference in the background so it overlaps this chunk's tail.
                if pending is None and cfg.lead_ticks > 0 and idx >= horizon - cfg.lead_ticks:
                    t_obs = time.monotonic()
                    ctx, _, _ = self._current_ctx_and_state()
                    pending = pool.submit(self._client.predict, ctx)

                # (3) execute one tick, re-anchored to the latest measurement.
                st = self._runtime.get_state()
                q_meas, dq_meas = st["q"], st["dq"]

                if cfg.watchdog_enabled and self._watchdog.update(dq_meas):
                    raise OscillationTripped("sustained oscillation on measured dq")

                target8 = self._adapter.row_to_joint_target(chunk[idx], q_meas)
                idx += 1
                smoothed = self._smoother.step(target8[:7])
                q_tgt = q_meas + np.clip(smoothed - q_meas, -cfg.max_step_rad, cfg.max_step_rad)
                q_tgt = clip_to_joint_limits(q_tgt)
                self._runtime.set_joint_target(q_tgt)
                if self._gripper_runtime is not None:
                    self._gripper_runtime.set_target(float(target8[7]))

                with self._lock:
                    self._telemetry = Telemetry(
                        state=State.RUNNING, q=q_meas.tolist(), dq=dq_meas.tolist(),
                        ee_pos=ee_pose_to_matrix(st["ee_pose"])[:3, 3].tolist(),
                        control_command_success_rate=st["control_command_success_rate"],
                        last_predict_ms=self._client.last_predict_ms,
                        n_replans=n_replans, n_late=n_late,
                    )

                t_next += dt
                sleep_s = t_next - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -dt:
                    t_next = time.monotonic()  # badly behind -- resync instead of racing to catch up
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._telemetry = Telemetry(state=State.ERROR, error=str(e))
            self._set_state(State.ERROR)
            if self._runtime is not None:
                self._runtime.stop()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
