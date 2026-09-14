"""HTTP client to an arbitrary policy server, built from a user-authored
RequestSpec instead of a hardcoded JSON shape.

Only does the network round trip + payload assembly. The async
chunk-overlap scheduling (firing the next /predict before the current chunk
runs out, dropping stale leading indices) lives in control_loop.py, ported
from ``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py``
(same reasoning: ZMQ/robot reads must stay on the control thread, so the
timing state doesn't belong inside this network-only class).

Two things live here that schema/sources.py deliberately doesn't handle,
because they're transport/session bookkeeping, not observation data (found
necessary against a real server, RoleVLA's serve_real_robot.py, which
requires a session_id from /reset echoed into every /predict along with a
strictly-increasing sequence number and a fresh request_id):
  - "session:id" / "session:sequence" / "session:request_id" /
    "session:timestamp_ns" sources, resolved from THIS client's own state.
  - "$ENV_VAR" substitution in ConnectionSpec.headers, resolved fresh per
    request so a secret (e.g. a bearer token) is never persisted resolved.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Optional

import numpy as np
import requests

from franka_deploy.schema.encode import build_field_value
from franka_deploy.schema.sources import SourceContext, resolve_source
from franka_deploy.schema.spec import RequestFieldSpec, RequestSpec

_ENV_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _resolve_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), v) for k, v in headers.items()}


def _assign_nested(payload: dict, name: str, value: Any) -> None:
    """"images.agentview_rgb" -> payload["images"]["agentview_rgb"] = value."""
    parts = name.split(".")
    d = payload
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


class PolicyClient:
    def __init__(self, spec: RequestSpec, timeout: float = 60.0):
        self.spec = spec
        self.timeout = timeout
        self.session = requests.Session()
        self.last_predict_ms: float = 0.0
        # Session bookkeeping for servers that require it (see module docstring).
        # None/0 are harmless no-ops for servers that don't use "session:*" sources.
        self.session_id: Optional[str] = None
        self._sequence = 0

    def _resolve_session_source(self, kind: str) -> Any:
        if kind == "id":
            return self.session_id
        if kind == "sequence":
            value = self._sequence
            self._sequence += 1
            return value
        if kind == "request_id":
            return uuid.uuid4().hex
        if kind == "timestamp_ns":
            return time.time_ns()
        raise ValueError(f"unknown session source 'session:{kind}'")

    def _build_body(self, fields: list[RequestFieldSpec], ctx: SourceContext) -> dict:
        payload: dict = {}
        for f in fields:
            if f.source.startswith("session:"):
                raw = self._resolve_session_source(f.source.split(":", 1)[1])
            else:
                raw = resolve_source(f.source, ctx)
            _assign_nested(payload, f.name, build_field_value(raw, f))
        return payload

    def build_payload(self, ctx: SourceContext) -> dict:
        return self._build_body(self.spec.fields, ctx)

    def _post(self, path: str, body: dict, timeout: Optional[float] = None) -> dict:
        url = self.spec.connection.base_url + path
        headers = _resolve_headers(self.spec.connection.headers)
        r = self.session.post(url, json=body, headers=headers, timeout=timeout or self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def reset(self, instruction: Optional[str] = None) -> dict:
        empty_ctx = SourceContext(cameras={}, robot_state={})
        body = self._build_body(self.spec.reset_fields, empty_ctx)
        if self.spec.instruction_field and instruction is not None:
            _assign_nested(body, self.spec.instruction_field, instruction)
        self._sequence = 0
        result = self._post(self.spec.connection.reset_endpoint, body)
        self.session_id = result.get("session_id")  # no-op if the server doesn't use sessions
        return result

    def predict(self, ctx: SourceContext) -> np.ndarray:
        """Returns the raw [T, D] response chunk, untouched (no action-space
        interpretation here -- that's schema/response_detect.py + apply.py,
        which need the confirmed spec this class doesn't have)."""
        payload = self.build_payload(ctx)
        t0 = time.perf_counter()
        data = self._post(self.spec.connection.predict_endpoint, payload)
        self.last_predict_ms = (time.perf_counter() - t0) * 1000
        if self.spec.actions_key not in data:
            raise KeyError(
                f"policy server response has no {self.spec.actions_key!r} key "
                f"(actual keys: {list(data)}) -- check RequestSpec.actions_key"
            )
        return np.asarray(data[self.spec.actions_key], dtype=float)
