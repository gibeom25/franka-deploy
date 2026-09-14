"""dict <-> dataclass conversion for the schema types, used by the config
API and by loading a saved YAML profile. Deliberately hand-written instead
of a generic (de)serialization library -- the schema dataclasses are flat
enough (spec.py) that this is a handful of lines and stays obvious."""

from __future__ import annotations

from dataclasses import asdict

from franka_deploy.schema.spec import (
    ActionSpaceSpec,
    ConnectionSpec,
    RequestFieldSpec,
    RequestSpec,
)


def request_spec_to_dict(spec: RequestSpec) -> dict:
    return asdict(spec)


def request_spec_from_dict(d: dict) -> RequestSpec:
    conn = ConnectionSpec(**d["connection"])
    fields = [RequestFieldSpec(**f) for f in d.get("fields", [])]
    return RequestSpec(
        connection=conn,
        fields=fields,
        actions_key=d.get("actions_key", "actions"),
        instruction_field=d.get("instruction_field"),
    )


def action_space_spec_to_dict(spec: ActionSpaceSpec) -> dict:
    return asdict(spec)


def action_space_spec_from_dict(d: dict) -> ActionSpaceSpec:
    return ActionSpaceSpec(**d)
