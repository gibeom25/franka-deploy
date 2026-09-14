"""Process-wide app state: one FR3 means one active session per process, so
a single ControlLoop (created lazily on first /api/control/connect) plus
the config that would be used to (re)build it is all this needs -- no
session registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from franka_deploy.cameras import CameraConfig
from franka_deploy.control_loop import ControlLoop, LoopConfig
from franka_deploy.safety.limits import DEFAULT_A_MAX, DEFAULT_FILTER_WN, DEFAULT_J_MAX, DEFAULT_V_MAX
from franka_deploy.schema.spec import RequestSpec


@dataclass
class AppConfig:
    robot_ip: str = "172.16.0.2"
    cameras: list[CameraConfig] = field(default_factory=list)
    request_spec: Optional[RequestSpec] = None
    loop: LoopConfig = field(default_factory=LoopConfig)
    v_max: float = DEFAULT_V_MAX
    a_max: float = DEFAULT_A_MAX
    j_max: float = DEFAULT_J_MAX
    filter_wn: float = DEFAULT_FILTER_WN


class AppState:
    def __init__(self) -> None:
        self.config = AppConfig()
        self.loop: Optional[ControlLoop] = None


APP_STATE = AppState()
