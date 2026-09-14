"""Tests the actual ZMQ transport between robot_node.py (the process that
will own FR3Runtime) and zmq_client.py (what control_loop.py talks to) --
no pylibfranka or hardware needed, FR3Runtime itself is monkeypatched out.
Everything else (pickle encode/decode, REQ/REP round trip, error
propagation, timeout-on-no-server) is real.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import franka_deploy.robot.robot_node as robot_node_module  # noqa: E402
from franka_deploy.robot.robot_node import serve  # noqa: E402
from franka_deploy.robot.zmq_client import RobotNodeError, ZMQRobotClient  # noqa: E402


class FakeFR3Runtime:
    """Stands in for the pylibfranka-backed FR3Runtime inside robot_node.py."""

    def __init__(self, robot_ip: str, read_only: bool = True, **kwargs):
        self.read_only = read_only
        self._q = np.zeros(7)
        self._dq = np.zeros(7)

    def get_state(self) -> dict:
        return {"q": self._q.copy(), "dq": self._dq.copy(),
                "ee_pose": np.eye(4).flatten(order="F"), "control_command_success_rate": 1.0}

    def set_joint_target(self, q) -> None:
        self._q = np.asarray(q, dtype=float).copy()

    def stop(self, *a, **k) -> None:
        pass


def _find_free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def running_node(monkeypatch):
    monkeypatch.setattr(robot_node_module, "FR3Runtime", FakeFR3Runtime)
    port = _find_free_port()
    thread = threading.Thread(target=serve, args=("172.16.0.2", port), daemon=True)
    thread.start()
    time.sleep(0.2)  # let the REP socket bind
    yield port


def test_connect_get_state_set_target_round_trip(running_node):
    client = ZMQRobotClient(f"tcp://127.0.0.1:{running_node}", timeout_ms=2000)
    result = client.connect(read_only=True)
    assert result == {"read_only": True}
    assert client.read_only is True

    state = client.get_state()
    assert np.allclose(state["q"], np.zeros(7))
    assert state["ee_pose"].shape == (16,)

    client.set_joint_target(np.full(7, 0.3))
    state2 = client.get_state()
    assert np.allclose(state2["q"], np.full(7, 0.3))

    client.stop()


def test_connect_live_mode_reflected(running_node):
    client = ZMQRobotClient(f"tcp://127.0.0.1:{running_node}", timeout_ms=2000)
    result = client.connect(read_only=False, v_max=0.5)
    assert result == {"read_only": False}
    assert client.read_only is False


def test_error_before_connect_propagates_as_robot_node_error(running_node):
    client = ZMQRobotClient(f"tcp://127.0.0.1:{running_node}", timeout_ms=2000)
    with pytest.raises(RobotNodeError, match="not connected"):
        client.get_state()


def test_no_server_times_out_instead_of_hanging():
    dead_port = _find_free_port()  # nothing listening here
    client = ZMQRobotClient(f"tcp://127.0.0.1:{dead_port}", timeout_ms=300)
    t0 = time.monotonic()
    with pytest.raises(RobotNodeError, match="did not respond"):
        client.connect(read_only=True)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0  # bounded by timeout_ms, not hanging forever


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
