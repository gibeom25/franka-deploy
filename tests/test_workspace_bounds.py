"""Unit tests for safety/workspace_bounds.py -- the teleop-recorded
Cartesian fence. No hardware needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from franka_deploy.safety.workspace_bounds import WorkspaceBounds, WorkspaceExceeded  # noqa: E402


def test_disabled_with_fewer_than_two_points():
    wb = WorkspaceBounds()
    assert not wb.enabled
    wb.add_point([0.4, 0.0, 0.3])
    assert not wb.enabled  # one point alone can't define a box


def test_check_is_noop_when_not_enabled():
    wb = WorkspaceBounds()
    wb.check([100.0, 100.0, 100.0])  # must not raise -- undefined bounds, nothing to enforce


def test_bbox_from_recorded_corners_with_margin():
    wb = WorkspaceBounds(margin=0.02)
    wb.add_point([0.2, -0.3, 0.1])
    wb.add_point([0.6, 0.3, 0.5])
    lo, hi = wb.bbox()
    assert np.allclose(lo, [0.22, -0.28, 0.12])
    assert np.allclose(hi, [0.58, 0.28, 0.48])


def test_inside_point_passes():
    wb = WorkspaceBounds(margin=0.0)
    wb.add_point([0.2, -0.3, 0.1])
    wb.add_point([0.6, 0.3, 0.5])
    wb.check([0.4, 0.0, 0.3])  # must not raise


def test_outside_point_raises():
    wb = WorkspaceBounds(margin=0.0)
    wb.add_point([0.2, -0.3, 0.1])
    wb.add_point([0.6, 0.3, 0.5])
    with pytest.raises(WorkspaceExceeded):
        wb.check([0.9, 0.0, 0.3])


def test_remove_point_and_clear():
    wb = WorkspaceBounds()
    wb.add_point([0.0, 0.0, 0.0])
    wb.add_point([1.0, 1.0, 1.0])
    wb.add_point([2.0, 2.0, 2.0])
    wb.remove_point(0)
    assert wb.points == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    wb.clear()
    assert wb.points == []
    assert not wb.enabled


def test_to_dict_shape():
    wb = WorkspaceBounds()
    d = wb.to_dict()
    assert d["enabled"] is False and d["lo"] is None and d["hi"] is None
    wb.add_point([0.0, 0.0, 0.0])
    wb.add_point([1.0, 1.0, 1.0])
    d = wb.to_dict()
    assert d["enabled"] is True
    assert len(d["lo"]) == 3 and len(d["hi"]) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
