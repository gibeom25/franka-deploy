"""Tests for the schema/client.py mechanisms added for a real server
(RoleVLA's serve_real_robot.py): nested "a.b" field names, "session:*"
sources (session_id/sequence/request_id/timestamp_ns), and "$ENV_VAR"
substitution in ConnectionSpec.headers. No hardware needed -- a local stub
HTTP server plays the policy server.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from franka_deploy.schema.client import PolicyClient, _assign_nested, _resolve_headers  # noqa: E402
from franka_deploy.schema.sources import SourceContext  # noqa: E402
from franka_deploy.schema.spec import ConnectionSpec, RequestFieldSpec, RequestSpec  # noqa: E402


def test_assign_nested_builds_dict_path():
    payload = {}
    _assign_nested(payload, "images.agentview_rgb", "AAA")
    _assign_nested(payload, "images.eye_in_hand_rgb", "BBB")
    _assign_nested(payload, "protocol_version", 1)
    assert payload == {"images": {"agentview_rgb": "AAA", "eye_in_hand_rgb": "BBB"}, "protocol_version": 1}


def test_resolve_headers_substitutes_env_var(monkeypatch):
    monkeypatch.setenv("ROLEVLA_API_TOKEN", "secret-token-value")
    headers = _resolve_headers({"Authorization": "Bearer $ROLEVLA_API_TOKEN", "X-Plain": "no-var-here"})
    assert headers == {"Authorization": "Bearer secret-token-value", "X-Plain": "no-var-here"}


def test_resolve_headers_missing_env_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    headers = _resolve_headers({"X": "$MISSING_VAR"})
    assert headers == {"X": ""}


class _CapturingHandler(BaseHTTPRequestHandler):
    captured = []

    def _read(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        return body

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self._read()
        _CapturingHandler.captured.append((self.path, dict(self.headers), body))
        if self.path == "/reset":
            self._send({"protocol_version": 1, "session_id": "sess-123", "task_id": body.get("task_id")})
        elif self.path == "/predict":
            self._send({"actions": [[0.0] * 8] * 10})
        else:
            self._send({}, status=404)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def stub_server():
    _CapturingHandler.captured = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()
    thread.join(timeout=2)


def test_session_bookkeeping_and_nested_images_and_auth_header(stub_server, monkeypatch):
    monkeypatch.setenv("ROLEVLA_API_TOKEN", "tok-abc")
    spec = RequestSpec(
        connection=ConnectionSpec(
            server_ip="127.0.0.1", server_port=stub_server,
            headers={"Authorization": "Bearer $ROLEVLA_API_TOKEN"},
        ),
        reset_fields=[
            RequestFieldSpec(name="protocol_version", source="static:1"),
        ],
        instruction_field="task_id",
        fields=[
            RequestFieldSpec(name="protocol_version", source="static:1"),
            RequestFieldSpec(name="session_id", source="session:id"),
            RequestFieldSpec(name="sequence", source="session:sequence"),
            RequestFieldSpec(name="request_id", source="session:request_id"),
            RequestFieldSpec(name="observation_timestamp_ns", source="session:timestamp_ns"),
            RequestFieldSpec(name="images.agentview_rgb", source="static:fake-png-b64"),
        ],
        actions_key="actions",
    )
    client = PolicyClient(spec)
    client.reset("some_task_id")
    assert client.session_id == "sess-123"

    ctx = SourceContext(cameras={}, robot_state={})
    client.predict(ctx)
    client.predict(ctx)  # second call -> sequence must have advanced

    reset_path, reset_headers, reset_body = _CapturingHandler.captured[0]
    assert reset_path == "/reset"
    assert reset_body == {"protocol_version": 1, "task_id": "some_task_id"}
    assert reset_headers["Authorization"] == "Bearer tok-abc"

    _, _, predict1 = _CapturingHandler.captured[1]
    _, _, predict2 = _CapturingHandler.captured[2]
    assert predict1["protocol_version"] == 1
    assert predict1["session_id"] == "sess-123"
    assert predict1["sequence"] == 0
    assert predict2["sequence"] == 1  # strictly increasing across calls
    assert predict1["images"] == {"agentview_rgb": "fake-png-b64"}
    assert isinstance(predict1["request_id"], str) and len(predict1["request_id"]) > 0
    assert predict1["request_id"] != predict2["request_id"]
    assert isinstance(predict1["observation_timestamp_ns"], int)


def test_static_source_parses_json_scalars():
    from franka_deploy.schema.sources import resolve_source

    ctx = SourceContext(cameras={}, robot_state={})
    assert resolve_source("static:1", ctx) == 1
    assert resolve_source("static:true", ctx) is True
    assert resolve_source("static:pick up the cup", ctx) == "pick up the cup"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
