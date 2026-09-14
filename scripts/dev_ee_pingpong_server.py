"""Second dev/test policy server: EE-delta "ping-pong" -- alternates a
small constant left/right (Y-axis, tool frame) position delta every
half_period seconds, zero rotation delta, gripper held open. Rotation
representation is axis_angle (3), matching the vendored, already-validated
``ee_step_to_joint`` convention from manipulation-stack (normalized delta
scaled by pos_max/rot_max, re-anchored to the measured EE pose every tick
by franka-deploy's control loop) -- see
franka_deploy/kinematics/fr3_kinematics.py.

Response is deliberately a NORMALIZED signal (roughly in [-1, 1]), not raw
meters -- this is realistic (matches how the one real EE-delta checkpoint
on this robot was trained) but means franka-deploy's auto-detector can't
reliably classify it by magnitude alone (documented limitation in
schema/response_detect.py / ActionSpaceSpec.delta_pos_scale). Confirm this
one manually via /api/control/confirm instead of relying on /detect:

    {"kind": "ee_delta", "dim": 7, "has_gripper": true,
     "gripper_convention": "-1_1", "rotation_repr": "axis_angle",
     "delta_pos_scale": 0.02, "delta_rot_scale": 0.0}

Usage:
    ~/pylibfranka-venv/bin/python scripts/dev_ee_pingpong_server.py \
        [--port 9000] [--half-period 2.5]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(half_period: float, chunk_len: int, dt: float):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/reset":
                self._send({"instruction": ""})
                return

            if self.path == "/predict":
                t0 = time.monotonic()
                chunk = []
                for i in range(chunk_len):
                    t = t0 + i * dt
                    phase = int(t // half_period) % 2
                    dy = 1.0 if phase == 0 else -1.0
                    chunk.append([0.0, dy, 0.0, 0.0, 0.0, 0.0, -1.0])  # dx,dy,dz,drx,dry,drz,gripper(-1=open)
                self._send({"actions": chunk})
                return

            self._send({}, status=404)

        def log_message(self, fmt, *args):
            print("[dev-ee-pingpong]", fmt % args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--half-period", type=float, default=2.5, help="seconds per left/right leg")
    ap.add_argument("--chunk-len", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05, help="seconds per chunk step (matches 20 Hz)")
    args = ap.parse_args()

    handler = make_handler(args.half_period, args.chunk_len, args.dt)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"[dev-ee-pingpong] half_period={args.half_period}s, listening on :{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
