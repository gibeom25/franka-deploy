"""End-to-end dry run against a local stub policy server implementing the
user-specified example protocol (configs/examples/libero_style_example.yaml)
-- no robot or camera hardware needed. Exercises the full request builder
(schema/client.py + encode.py + sources.py, including a `custom:` plugin
for the language embedding) and the response auto-detector
(response_detect.py) together, the way control_loop.py actually chains
them. This is the automated version of manipulation-stack's
``fr3_policy_client.py --dry-run`` (synthetic obs, real HTTP round trip).
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from franka_deploy.schema.client import PolicyClient  # noqa: E402
from franka_deploy.schema.response_detect import detect_action_space  # noqa: E402
from franka_deploy.schema.serialize import request_spec_from_dict  # noqa: E402
from franka_deploy.schema.sources import SourceContext, robot_state_dict  # noqa: E402
from franka_deploy.schema.spec import ACTION_SPACE_JOINT_ABSOLUTE  # noqa: E402

RESET_Q = np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.785398])
EXAMPLE_PATH = (
    Path(__file__).resolve().parent.parent / "franka_deploy" / "configs" / "examples" / "libero_style_example.yaml"
)


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
            # Sanity-check the request actually matches the example schema
            # (proves the client, not just the server, honors the spec).
            assert "agentview_image" in body and "eye_in_hand_image" in body and "lang_emb" in body
            img = np.array(body["agentview_image"])
            assert img.shape == (1, 1, 3, 128, 128)
            lang = np.array(body["lang_emb"])
            assert lang.shape == (1, 512)
            chunk = np.tile(np.append(RESET_Q + 0.01, 0.4), (10, 1)).tolist()
            self._send_json({"actions": chunk})
        else:
            self._send_json({}, status=404)

    def log_message(self, fmt, *args):  # noqa: A002 -- silence per-request logging
        pass


def test_full_request_response_round_trip_against_stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = yaml.safe_load(EXAMPLE_PATH.read_text())
        spec = request_spec_from_dict(cfg["request_spec"])
        spec.connection.server_ip = "127.0.0.1"
        spec.connection.server_port = port

        client = PolicyClient(spec)
        client.reset("pick up the cup")

        rng = np.random.default_rng(0)
        cameras = {
            "agentview": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "eye_in_hand": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        robot_state = robot_state_dict(
            RESET_Q, np.zeros(7), np.array([0.45, 0.0, 0.35]), np.array([0.0, 0.0, 0.0, 1.0]), 0.0
        )
        ctx = SourceContext(cameras=cameras, robot_state=robot_state)

        chunk = client.predict(ctx)
        assert chunk.shape == (10, 8)

        result = detect_action_space(chunk, RESET_Q, np.array([0.45, 0.0, 0.35]))
        assert result.spec.kind == ACTION_SPACE_JOINT_ABSOLUTE
        assert result.spec.has_gripper
    finally:
        server.shutdown()
        thread.join(timeout=2)


if __name__ == "__main__":
    test_full_request_response_round_trip_against_stub_server()
    print("OK")
