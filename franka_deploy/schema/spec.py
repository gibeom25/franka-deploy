"""Declarative spec for what franka-deploy sends to / expects from a policy server.

This is the generalization layer the whole project exists for: instead of a
client hardcoded to one server's JSON shape (see
``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py``, which
only ever sends ``observation.state``/``observation.images.agent`` etc.),
the user writes a spec like this and any model's protocol works. Nothing
here talks to the network or the robot -- see ``client.py`` for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# ---------------------------------------------------------------- request

# Where a field's raw value comes from before encoding.
#   "camera:<role>"   -> latest RGB frame from that camera role (see cameras/)
#   "robot:<key>"     -> current robot state (see schema/sources.py ROBOT_STATE_KEYS)
#   "static:<value>"  -> a fixed value the user typed in (e.g. an instruction string)
#   "custom:<dotted.path>" -> a user plugin function (see plugins.py)
SourceKind = Literal["camera", "robot", "static", "custom"]

Encoding = Literal["none", "raw_bytes_base64", "png_base64", "jpeg_base64"]


@dataclass
class RequestFieldSpec:
    """One key in the outbound JSON request body.

    Example (image field, matches a user-supplied protocol where the model
    wants a resized float32 CHW tensor):
        RequestFieldSpec(
            name="agentview_image", source="camera:agentview",
            dtype="float32", shape=[1, 1, 3, 128, 128], resize=[128, 128],
            layout="CHW", normalize={"scale": 1.0/255.0},
        )
    Example (raw bytes, matches manipulation-stack's existing protocol):
        RequestFieldSpec(
            name="observation.images.agent", source="camera:agentview",
            dtype="uint8", encoding="raw_bytes_base64",
        )
    Example (server-side embedding not needed, just pass the instruction):
        RequestFieldSpec(name="instruction", source="static:pick up the cup")
    Example (something no generic pipeline can produce, e.g. a language
    embedding computed locally):
        RequestFieldSpec(name="lang_emb", source="custom:my_plugins.embed_instruction")
    """

    name: str
    source: str  # "<kind>:<detail>", parsed by schema/sources.py
    dtype: Optional[str] = None  # numpy dtype name, e.g. "uint8", "float32"
    shape: Optional[list[int]] = None  # target shape after resize/reshape, e.g. [1,1,3,128,128]
    resize: Optional[list[int]] = None  # [H, W] for image fields; None = no resize
    layout: Literal["HWC", "CHW"] = "HWC"
    normalize: Optional[dict[str, float]] = None  # {"mean":.., "std":..} or {"scale":..}
    encoding: Encoding = "none"  # how the final array is put into JSON
    transform_fn: Optional[str] = None  # dotted path, applied AFTER encoding/shaping (rare)


@dataclass
class ConnectionSpec:
    server_ip: str
    server_port: int = 8080
    scheme: str = "http"
    predict_endpoint: str = "/predict"
    reset_endpoint: str = "/reset"

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.server_ip}:{self.server_port}"


@dataclass
class RequestSpec:
    connection: ConnectionSpec
    fields: list[RequestFieldSpec] = field(default_factory=list)
    actions_key: str = "actions"  # JSON key in the /predict response holding the [T,D] chunk
    instruction_field: Optional[str] = None  # name of a field carrying the reset instruction, if any


# ---------------------------------------------------------------- response

ACTION_SPACE_JOINT_ABSOLUTE = "joint_absolute"
ACTION_SPACE_JOINT_DELTA = "joint_delta"
ACTION_SPACE_EE_ABSOLUTE = "ee_absolute"
ACTION_SPACE_EE_DELTA = "ee_delta"

GripperConvention = Literal["0_1", "-1_1", "none"]


@dataclass
class ActionSpaceSpec:
    """What one row of the response chunk means. Produced by auto-detection
    (schema/response_detect.py) and confirmed/overridden by the user before
    any motion is commanded."""

    kind: str  # one of the ACTION_SPACE_* constants
    dim: int  # total columns in one response row, including gripper if present
    has_gripper: bool
    gripper_convention: GripperConvention = "0_1"
    rotation_repr: Optional[str] = None  # "quat" | "rot6d" | "axis_angle" | None (joint modes)
    confidence: float = 1.0  # 0..1, how sure response_detect.py was
    notes: str = ""
    # Delta scaling: multiplies the raw response columns before they're
    # added to the current pose (schema/apply.py). Default 1.0 = the server
    # already sends raw radians/meters. Some protocols instead send a
    # normalized signal in roughly [-1, 1] (e.g. manipulation-stack's
    # EE-delta convention, pos_max=0.05 m / rot_max=0.5 rad) -- if so, set
    # these to that model's actual scale. response_detect.py cannot infer
    # this on its own; the user sets it when confirming a *_delta detection.
    delta_pos_scale: float = 1.0
    delta_rot_scale: float = 1.0
    delta_joint_scale: float = 1.0


@dataclass
class DetectionResult:
    spec: ActionSpaceSpec
    sample_row: list[float]
    chunk_shape: tuple[int, int]
    candidates: list[ActionSpaceSpec] = field(default_factory=list)
