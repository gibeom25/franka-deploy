"""Robot node: owns the 1 kHz FR3Runtime control loop in its OWN OS
process, exposed to the app process (FastAPI, IK, policy HTTP client,
cameras) over a ZMQ REQ/REP socket.

Why this exists: two real hardware faults were observed running
everything in one process --
  - `communication_constraints_violation` right after connect (web/HTTP
    setup work contending with the control thread's startup)
  - `joint_motion_generator_velocity/acceleration_discontinuity` during
    EE-delta actions (analytic IK, ~4ms/call, holding the GIL long enough
    to starve the 1 kHz thread)
A GIL-tuning workaround (schedwitch interval, reordering start()) fixed
the first but not the second -- IK cost doesn't go away. The actual fix is
this process boundary: nothing CPU-heavy ever shares a GIL with the
control thread again. This mirrors manipulation-stack's own
`mstack.comm.zmq_core.robot_node.ZMQServerRobot`, already validated in
production for the same reason.

Run (pylibfranka-venv ONLY -- nothing else in this project needs pylibfranka):
    ~/pylibfranka-venv/bin/python -m franka_deploy.robot.robot_node \
        --robot-ip 172.16.0.2 --port 5560

Protocol: ZMQ REQ/REP, pickle-encoded requests/responses (numpy arrays
don't survive JSON round trips without extra work -- pickle does, and
this socket is loopback-only, never exposed to the untrusted policy
server). Single client at a time, matching the FCI's own one-client limit.
"""

from __future__ import annotations

import argparse
import pickle
import traceback
from typing import Any, Optional

import zmq

from franka_deploy.robot.fr3_runtime import FR3Runtime


class RobotNode:
    """Dispatch target for incoming RPCs -- one FR3Runtime at a time,
    (re)created on each `connect()` call (mirrors ControlLoop.connect()'s
    old in-process behavior: stop the previous one, open a fresh one)."""

    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self.runtime: Optional[FR3Runtime] = None

    def connect(self, read_only: bool = True, **safety_kwargs: Any) -> dict:
        if self.runtime is not None:
            self.runtime.stop()
        kwargs = {k: v for k, v in safety_kwargs.items() if v is not None}
        self.runtime = FR3Runtime(robot_ip=self.robot_ip, read_only=read_only, **kwargs)
        return {"read_only": self.runtime.read_only}

    def get_state(self) -> dict:
        if self.runtime is None:
            raise RuntimeError("not connected -- call connect() first")
        return self.runtime.get_state()

    def set_joint_target(self, q) -> None:
        if self.runtime is None:
            raise RuntimeError("not connected -- call connect() first")
        self.runtime.set_joint_target(q)

    def read_only(self) -> bool:
        if self.runtime is None:
            raise RuntimeError("not connected -- call connect() first")
        return self.runtime.read_only

    def stop(self) -> None:
        if self.runtime is not None:
            self.runtime.stop()

    def disconnect(self) -> None:
        if self.runtime is not None:
            self.runtime.stop()
            self.runtime = None


def serve(robot_ip: str, port: int) -> None:
    node = RobotNode(robot_ip)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")
    print(f"[robot-node] robot_ip={robot_ip}, listening on :{port}")
    while True:
        raw = sock.recv()
        try:
            req = pickle.loads(raw)
            method = req["method"]
            args = req.get("args", {})
            result = getattr(node, method)(**args)
            sock.send(pickle.dumps(result))
        except Exception as e:  # noqa: BLE001 -- must always reply, REQ/REP hangs forever otherwise
            traceback.print_exc()
            sock.send(pickle.dumps({"__error__": f"{type(e).__name__}: {e}"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="172.16.0.2")
    ap.add_argument("--port", type=int, default=5560)
    args = ap.parse_args()
    serve(args.robot_ip, args.port)


if __name__ == "__main__":
    main()
