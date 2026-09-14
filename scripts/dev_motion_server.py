"""Minimal local policy server for a first real-motion smoke test of
franka-deploy's full pipeline -- NOT a real policy, just a predictable,
bounded motion generator so the pipeline (schema send -> detect -> confirm
-> execute) can be validated on real hardware before pointing at an
actual model.

Behavior: joint_absolute, holds the client's observed pose and adds a
small, slow sinusoidal wiggle to joint 7 only (index 6 -- wide range
[-3.02, 3.16] rad, and rotating the wrist has low Cartesian excursion
compared to the same amplitude on an elbow/shoulder joint). Gripper stays
at whatever the client reports (held open by default).

Usage:
    ~/pylibfranka-venv/bin/python scripts/dev_motion_server.py \
        [--port 9000] [--amplitude 0.1] [--period 4.0]

Pair with franka_deploy/configs/examples/dev_motion_test.yaml, which
points franka-deploy at this server with a minimal request schema (just
the current joint state, no cameras needed).
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def make_handler(amplitude: float, period: float, chunk_len: int, dt: float):
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
            body = json.loads(self.rfile.read(length)) if length else {}

            if self.path == "/reset":
                self._send({"instruction": body.get("instruction", "")})
                return

            if self.path == "/predict":
                state = body.get("state")
                if not state or len(state) < 7:
                    self._send({"error": "missing/short 'state' field (need >=7 joints)"}, status=400)
                    return
                q = np.array(state[:7], dtype=float)
                gripper = float(state[7]) if len(state) > 7 else 0.0
                t0 = time.monotonic()
                chunk = []
                for i in range(chunk_len):
                    t = t0 + i * dt
                    wiggle = amplitude * np.sin(2 * np.pi * t / period)
                    qi = q.copy()
                    qi[6] += wiggle
                    chunk.append(np.append(qi, gripper).tolist())
                self._send({"actions": chunk})
                return

            self._send({}, status=404)

        def log_message(self, fmt, *args):
            print("[dev-motion-server]", fmt % args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--amplitude", type=float, default=0.1, help="rad, joint7 wiggle amplitude")
    ap.add_argument("--period", type=float, default=4.0, help="seconds per full wiggle cycle")
    ap.add_argument("--chunk-len", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.05, help="seconds per chunk step (matches 20 Hz)")
    args = ap.parse_args()

    handler = make_handler(args.amplitude, args.period, args.chunk_len, args.dt)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"[dev-motion-server] joint7 wiggle amplitude={args.amplitude} rad, "
          f"period={args.period}s, listening on :{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
