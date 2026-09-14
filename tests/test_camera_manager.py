"""Unit tests for cameras/manager.py's background-threaded reader --
no hardware needed (MockCamera stands in for RealSenseCamera)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from franka_deploy.cameras.manager import CameraManager  # noqa: E402
from franka_deploy.cameras.mock import MockCamera  # noqa: E402


def test_start_populates_latest_frame_without_manual_read():
    mgr = CameraManager()
    try:
        mgr.start({"agentview": MockCamera(width=32, height=24)})
        deadline = time.monotonic() + 2.0
        rgb, error = None, None
        while time.monotonic() < deadline:
            rgb, error = mgr.get_latest("agentview")
            if rgb is not None:
                break
            time.sleep(0.01)
        assert error is None
        assert rgb is not None
        assert rgb.shape == (24, 32, 3)
    finally:
        mgr.stop()


def test_unknown_role_reports_error_not_exception():
    mgr = CameraManager()
    try:
        mgr.start({"agentview": MockCamera()})
        rgb, error = mgr.get_latest("eye_in_hand")
        assert rgb is None
        assert "not started" in error
    finally:
        mgr.stop()


def test_restart_replaces_previous_cameras():
    mgr = CameraManager()
    try:
        mgr.start({"agentview": MockCamera()})
        assert mgr.roles() == ["agentview"]
        mgr.start({"eye_in_hand": MockCamera()})
        assert mgr.roles() == ["eye_in_hand"]
        rgb, error = mgr.get_latest("agentview")
        assert rgb is None
        assert error is not None
    finally:
        mgr.stop()


def test_stop_clears_roles():
    mgr = CameraManager()
    mgr.start({"agentview": MockCamera()})
    mgr.stop()
    assert mgr.roles() == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
