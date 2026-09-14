from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    role: str
    serial: Optional[str] = None  # None -> MockCamera (no hardware needed)
    width: int = 640
    height: int = 480
    fps: int = 30


def build_cameras(configs: list[CameraConfig]) -> dict:
    """role -> camera object exposing .read() -> (rgb HWC uint8, depth)."""
    from franka_deploy.cameras.mock import MockCamera
    from franka_deploy.cameras.realsense import RealSenseCamera

    cams = {}
    for c in configs:
        if c.serial:
            cams[c.role] = RealSenseCamera(c.serial, c.width, c.height, c.fps)
        else:
            cams[c.role] = MockCamera(c.width, c.height)
    return cams
