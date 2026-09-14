"""App-process-side stub for robot_node.py. Matches FR3Runtime's public
surface (read_only, get_state, set_joint_target, stop) plus a `connect()`
method, so control_loop.py talks to the robot the same way whether it's
local or remote -- it just no longer imports pylibfranka at all.
"""

from __future__ import annotations

import pickle
from typing import Any

import zmq


class RobotNodeError(RuntimeError):
    """Raised when the robot node doesn't reply (not running, or crashed)
    or returns an error for the requested RPC."""


class ZMQRobotClient:
    def __init__(self, address: str = "tcp://127.0.0.1:5560", timeout_ms: int = 5000):
        self._address = address
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._read_only = True
        self._open_socket()

    def _open_socket(self) -> None:
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self._address)

    def _call(self, method: str, **kwargs: Any) -> Any:
        try:
            self._sock.send(pickle.dumps({"method": method, "args": kwargs}))
            raw = self._sock.recv()
        except zmq.error.Again as e:
            # A REQ socket that times out is stuck mid-cycle (can't send
            # again without a reply) -- must be recreated before any next call.
            self._sock.close(0)
            self._open_socket()
            raise RobotNodeError(
                f"robot node at {self._address} did not respond within {self._timeout_ms}ms "
                f"to {method!r} -- is `python -m franka_deploy.robot.robot_node` running?"
            ) from e
        result = pickle.loads(raw)
        if isinstance(result, dict) and "__error__" in result:
            raise RobotNodeError(result["__error__"])
        return result

    def connect(self, read_only: bool = True, **safety_kwargs: Any) -> dict:
        result = self._call("connect", read_only=read_only, **safety_kwargs)
        self._read_only = result["read_only"]
        return result

    @property
    def read_only(self) -> bool:
        return self._read_only

    def get_state(self) -> dict:
        return self._call("get_state")

    def set_joint_target(self, q) -> None:
        self._call("set_joint_target", q=q)

    def stop(self, *args: Any, **kwargs: Any) -> None:
        # *args/**kwargs accepted-and-ignored: FR3Runtime.stop() takes
        # decel_timeout/decel_vel_tol locally; the node applies its own
        # defaults, callers here only ever need "stop the robot."
        self._call("stop")

    def disconnect(self) -> None:
        self._call("disconnect")
