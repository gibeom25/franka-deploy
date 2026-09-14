"""RealSense camera interface: D405 (eye_in_hand) + D455 (agentview).

Serials and roles come from ``configs/default.yaml`` (found while researching
``gello_software/gello/libero_gui_worker.py`` -- re-scan with
``get_device_ids()`` below if a camera is swapped and they stop matching).

``preprocess_libero`` reproduces ``gello_software/gello/libero_format.py``'s
``resize_rgb`` byte-for-byte (center-crop to square, ``cv2.resize`` with
``INTER_AREA``) -- a BC/VLA checkpoint trained on that dataset expects this
exact preprocessing at inference time, not an approximation of it.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np


def list_devices() -> List[dict]:
    """Enumerate connected RealSense devices with name + serial, for a
    dashboard picker -- just query_devices(), no pipeline opened, so it's
    safe to call even while another camera on the bus is mid-stream."""
    import pyrealsense2 as rs

    ctx = rs.context()
    return [
        {"name": dev.get_info(rs.camera_info.name), "serial": dev.get_info(rs.camera_info.serial_number)}
        for dev in ctx.query_devices()
    ]


def get_device_ids(reset: bool = False) -> List[str]:
    """List connected RealSense serials.

    ``reset=False`` (default) just enumerates -- safe to call right before
    opening pipelines for those same serials. ``reset=True`` power-cycles
    every device over USB first (useful to unstick a camera left streaming
    by a crashed process) and blocks ~2s for re-enumeration; do NOT open a
    pipeline immediately after without that settle time; RealSenseCamera's
    own start-retry below covers the rest of the race, but the device may
    simply not exist on the bus yet within the first second or so.
    """
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    ids = []
    for dev in devices:
        if reset:
            dev.hardware_reset()
        ids.append(dev.get_info(rs.camera_info.serial_number))
    if reset:
        time.sleep(2)
    return ids


def preprocess_libero(img: np.ndarray, size: int = 256) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (size, size, 3), center-cropped square first.

    Must match gello_software's libero_format.resize_rgb exactly -- this is
    the train/inference parity point for any BC/VLA checkpoint.
    """
    import cv2

    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    cropped = img[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


class RealSenseCamera:
    def __repr__(self) -> str:
        return f"RealSenseCamera(serial={self._serial})"

    def __init__(
        self, serial: str, width: int = 640, height: int = 480, fps: int = 30,
        start_retries: int = 5, retry_delay: float = 1.0,
    ):
        import pyrealsense2 as rs

        self._serial = serial
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # A device that was just hardware_reset() (or opened right after
        # another camera on a shared hub) can still be re-enumerating on the
        # USB bus for a beat -- pipeline.start() then raises "No device
        # connected" even though query_devices() already saw the serial.
        # Retry with backoff instead of failing on the first attempt.
        last_err: Optional[Exception] = None
        for attempt in range(start_retries):
            try:
                self._pipeline.start(config)
                return
            except RuntimeError as e:  # noqa: PERF203
                last_err = e
                if attempt < start_retries - 1:
                    time.sleep(retry_delay)
        raise RuntimeError(
            f"RealSenseCamera(serial={serial}): failed to start after {start_retries} attempts"
        ) from last_err

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (color RGB (H,W,3), depth (H,W,1))."""
        frames = self._pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())[:, :, ::-1]  # BGR->RGB
        depth = np.asanyarray(frames.get_depth_frame().get_data())[:, :, None]
        return color, depth

    def stop(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


class FR3Cameras:
    """Both cameras, keyed by the same role names the training dataset uses
    (``agentview``/``eye_in_hand``) so a policy adapter's obs dict lines up
    with ``dataset_schema.py``'s ``agentview_rgb``/``eye_in_hand_rgb`` keys."""

    def __init__(self, cam_config: dict, image_size: int = 256):
        self._image_size = image_size
        self._cams = {
            role: RealSenseCamera(cfg["serial"], cfg.get("width", 640), cfg.get("height", 480), cfg.get("fps", 30))
            for role, cfg in cam_config.items()
            if role != "image_size"
        }

    def read_all(self) -> dict:
        obs = {}
        for role, cam in self._cams.items():
            color, depth = cam.read()
            obs[f"{role}_rgb"] = color
            obs[f"{role}_rgb_256"] = preprocess_libero(color, self._image_size)
            obs[f"{role}_depth"] = depth
        return obs

    def stop(self) -> None:
        for cam in self._cams.values():
            cam.stop()
