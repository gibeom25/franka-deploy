"""HTTP client to an arbitrary policy server, built from a user-authored
RequestSpec instead of a hardcoded JSON shape.

Only does the network round trip + payload assembly. The async
chunk-overlap scheduling (firing the next /predict before the current chunk
runs out, dropping stale leading indices) lives in control_loop.py, ported
from ``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py``
(same reasoning: ZMQ/robot reads must stay on the control thread, so the
timing state doesn't belong inside this network-only class).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import requests

from franka_deploy.schema.encode import build_field_value
from franka_deploy.schema.sources import SourceContext, resolve_source
from franka_deploy.schema.spec import RequestSpec


class PolicyClient:
    def __init__(self, spec: RequestSpec, timeout: float = 60.0):
        self.spec = spec
        self.timeout = timeout
        self.session = requests.Session()
        self.last_predict_ms: float = 0.0

    def build_payload(self, ctx: SourceContext) -> dict:
        payload = {}
        for f in self.spec.fields:
            raw = resolve_source(f.source, ctx)
            payload[f.name] = build_field_value(raw, f)
        return payload

    def reset(self, instruction: Optional[str] = None) -> dict:
        body: dict = {}
        if self.spec.instruction_field and instruction is not None:
            body[self.spec.instruction_field] = instruction
        url = self.spec.connection.base_url + self.spec.connection.reset_endpoint
        r = self.session.post(url, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    def predict(self, ctx: SourceContext) -> np.ndarray:
        """Returns the raw [T, D] response chunk, untouched (no action-space
        interpretation here -- that's schema/response_detect.py + apply.py,
        which need the confirmed spec this class doesn't have)."""
        payload = self.build_payload(ctx)
        url = self.spec.connection.base_url + self.spec.connection.predict_endpoint
        t0 = time.perf_counter()
        r = self.session.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        self.last_predict_ms = (time.perf_counter() - t0) * 1000
        data = r.json()
        if self.spec.actions_key not in data:
            raise KeyError(
                f"policy server response has no {self.spec.actions_key!r} key "
                f"(actual keys: {list(data)}) -- check RequestSpec.actions_key"
            )
        return np.asarray(data[self.spec.actions_key], dtype=float)
