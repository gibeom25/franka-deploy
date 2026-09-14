"""Full state-machine test for control_loop.ControlLoop (connect -> detect ->
confirm -> start -> stop) against a stub HTTP policy server and a fake
robot-node client -- no pylibfranka, ZMQ, or hardware needed. This is the
piece of the project with the most moving parts (async chunk overlap,
staged rollout, threading) and the least coverage otherwise.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import franka_deploy.control_loop as control_loop_module  # noqa: E402
from franka_deploy.cameras.manager import CameraManager  # noqa: E402
from franka_deploy.cameras.mock import MockCamera  # noqa: E402
from franka_deploy.control_loop import ControlLoop, LoopConfig, State  # noqa: E402
from franka_deploy.schema.serialize import request_spec_from_dict  # noqa: E402
from franka_deploy.schema.spec import ACTION_SPACE_JOINT_ABSOLUTE  # noqa: E402

RESET_Q = np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398])
EE_POS_NOW = np.array([0.45, 0.0, 0.35])
EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent / "franka_deploy" / "configs" / "examples" / "libero_style_example.yaml"
)


class FakeRobotNodeClient:
    """Stands in for robot/zmq_client.ZMQRobotClient (i.e. the whole
    separate robot_node.py process + pylibfranka): instantly "tracks"
    whatever target it's given (no reference filter -- that's already
    covered by test_reference_filter.py) so the control loop's own timing
    and state-machine logic can be tested without hardware or a second
    process. Matches ZMQRobotClient's shape: constructed with just an
    address, `.connect(read_only=..., **safety_kwargs)` sets the mode."""

    def __init__(self, address: str):
        self.read_only = True
        self._lock = threading.Lock()
        self._q = RESET_Q.copy()
        self._dq = np.zeros(7)
        T = np.eye(4)
        T[:3, 3] = EE_POS_NOW
        self._ee_pose_flat = T.flatten(order="F")

    def connect(self, read_only: bool = True, **safety_kwargs) -> dict:
        self.read_only = read_only
        return {"read_only": read_only}

    def get_state(self) -> dict:
        with self._lock:
            return {
                "q": self._q.copy(), "dq": self._dq.copy(),
                "ee_pose": self._ee_pose_flat.copy(),
                "control_command_success_rate": 1.0,
            }

    def set_joint_target(self, q) -> None:
        with self._lock:
            q = np.asarray(q, dtype=float)
            self._dq = (q - self._q) / 0.02
            self._q = q.copy()

    def stop(self, *args, **kwargs) -> None:
        pass


class _StubHandler(BaseHTTPRequestHandler):
    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self._read_json()
        if self.path == "/reset":
            self._send_json({"instruction": body.get("instruction", "")})
        elif self.path == "/predict":
            chunk = np.tile(np.append(RESET_Q + 0.01, 0.4), (10, 1)).tolist()
            self._send_json({"actions": chunk})
        else:
            self._send_json({}, status=404)

    def log_message(self, fmt, *args):
        pass


def test_control_loop_full_staged_rollout(monkeypatch):
    monkeypatch.setattr(control_loop_module, "ZMQRobotClient", FakeRobotNodeClient)

    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = yaml.safe_load(EXAMPLE_PATH.read_text())
        spec = request_spec_from_dict(cfg["request_spec"])
        spec.connection.server_ip = "127.0.0.1"
        spec.connection.server_port = port

        camera_manager = CameraManager()
        camera_manager.start({"agentview": MockCamera(), "eye_in_hand": MockCamera()})
        loop_cfg = LoopConfig(fps=50.0, exec_horizon=10, lead_ticks=2)
        loop = ControlLoop("tcp://127.0.0.1:5560", camera_manager, spec, loop_cfg)

        loop.connect(read_only=True)
        assert loop.state == State.CONNECTED

        result = loop.detect(instruction="pick up the cup")
        assert loop.state == State.AWAITING_CONFIRM
        assert result.spec.kind == ACTION_SPACE_JOINT_ABSOLUTE
        assert result.spec.has_gripper

        loop.confirm(result.spec)
        assert loop.state == State.ARMED

        loop.start(instruction="pick up the cup")
        assert loop.state == State.RUNNING
        time.sleep(0.6)  # several replans at 50 Hz / exec_horizon=10 -> ~0.2s per replan

        telemetry = loop.get_telemetry()
        assert telemetry.state == State.RUNNING
        assert telemetry.n_replans >= 2
        assert telemetry.q is not None

        loop.stop()
        assert loop.state == State.STOPPED
    finally:
        camera_manager.stop()
        server.shutdown()
        thread.join(timeout=2)


def test_control_loop_stops_on_workspace_violation(monkeypatch):
    """A workspace fence that EXCLUDES the robot's actual (fake, fixed)
    measured position must stop the loop the same way OscillationTripped
    does -- proves the enforcement wiring in _run(), not just the
    WorkspaceBounds class in isolation (already covered by
    test_workspace_bounds.py)."""
    monkeypatch.setattr(control_loop_module, "ZMQRobotClient", FakeRobotNodeClient)

    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = yaml.safe_load(EXAMPLE_PATH.read_text())
        spec = request_spec_from_dict(cfg["request_spec"])
        spec.connection.server_ip = "127.0.0.1"
        spec.connection.server_port = port

        camera_manager = CameraManager()
        camera_manager.start({"agentview": MockCamera(), "eye_in_hand": MockCamera()})
        loop_cfg = LoopConfig(fps=50.0, exec_horizon=10, lead_ticks=2)
        loop = ControlLoop("tcp://127.0.0.1:5560", camera_manager, spec, loop_cfg)

        # A box far from EE_POS_NOW ([0.45, 0.0, 0.35]) -- the fake robot's
        # measured position will never be inside it.
        loop.workspace.add_point([1.0, 1.0, 1.0])
        loop.workspace.add_point([1.2, 1.2, 1.2])
        assert loop.workspace.enabled

        loop.connect(read_only=True)
        result = loop.detect(instruction="pick up the cup")
        loop.confirm(result.spec)
        loop.start(instruction="pick up the cup")

        deadline = time.monotonic() + 2.0
        while loop.state != State.ERROR and time.monotonic() < deadline:
            time.sleep(0.02)

        assert loop.state == State.ERROR
        telemetry = loop.get_telemetry()
        assert "workspace" in telemetry.error.lower()
    finally:
        camera_manager.stop()
        server.shutdown()
        thread.join(timeout=2)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
