"""Process-wide app state: one FR3 means one active session per process, so
a single ControlLoop (created lazily on first /api/control/connect) plus
the config that would be used to (re)build it is all this needs -- no
session registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from franka_deploy.cameras import CameraConfig
from franka_deploy.cameras.manager import CameraManager
from franka_deploy.control_loop import ControlLoop, LoopConfig
from franka_deploy.safety.limits import DEFAULT_A_MAX, DEFAULT_FILTER_WN, DEFAULT_J_MAX, DEFAULT_V_MAX
from franka_deploy.safety.workspace_bounds import WorkspaceBounds
from franka_deploy.schema.spec import RequestSpec


@dataclass
class AppConfig:
    # Address of the robot_node.py process (separate OS process -- the 1 kHz
    # control loop must never share a GIL with this app; see
    # franka_deploy/robot/robot_node.py), NOT the robot's own FCI IP -- that
    # lives in the node process's own --robot-ip launch argument.
    robot_node_address: str = "tcp://127.0.0.1:5560"
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
        self.camera_manager = CameraManager()
        # Lives outside ControlLoop and survives /disconnect on purpose --
        # a workspace fence recorded by hand-guiding the robot is a fact
        # about the physical setup, not about one session.
        self.workspace = WorkspaceBounds()

        # Restore whatever was last saved (see api/persist.py) so an app
        # restart doesn't silently wipe the operator's config or a
        # hand-recorded workspace fence.
        from franka_deploy.api import persist

        persist.load_config_into(self.config)
        persist.load_workspace_into(self.workspace)


APP_STATE = AppState()
