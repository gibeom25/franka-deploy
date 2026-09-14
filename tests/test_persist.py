"""Round-trip tests for api/persist.py -- config and workspace bounds must
survive being saved and reloaded (the whole point: an app restart no
longer wipes them). No hardware needed; disk I/O redirected into tmp_path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from franka_deploy.api import persist  # noqa: E402
from franka_deploy.api.state import AppConfig  # noqa: E402
from franka_deploy.cameras import CameraConfig  # noqa: E402
from franka_deploy.control_loop import LoopConfig  # noqa: E402
from franka_deploy.safety.workspace_bounds import WorkspaceBounds  # noqa: E402
from franka_deploy.schema.serialize import request_spec_from_dict  # noqa: E402

EXAMPLE_SPEC = {
    "connection": {"server_ip": "127.0.0.1", "server_port": 9000, "scheme": "http",
                   "predict_endpoint": "/predict", "reset_endpoint": "/reset"},
    "actions_key": "actions",
    "instruction_field": "instruction",
    "fields": [{"name": "state", "source": "robot:joint_positions", "dtype": "float32",
                "shape": None, "resize": None, "layout": "HWC", "normalize": None,
                "encoding": "none", "transform_fn": None}],
}


@pytest.fixture(autouse=True)
def redirect_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(persist, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(persist, "WORKSPACE_PATH", tmp_path / "workspace.json")
    monkeypatch.setattr(persist, "STATE_DIR", tmp_path)


def test_load_returns_false_when_nothing_saved():
    cfg = AppConfig()
    assert persist.load_config_into(cfg) is False
    ws = WorkspaceBounds()
    assert persist.load_workspace_into(ws) is False


def test_config_round_trip():
    cfg = AppConfig(
        robot_node_address="tcp://127.0.0.1:5560",
        cameras=[CameraConfig(role="agentview", serial="338122300664")],
        request_spec=request_spec_from_dict(EXAMPLE_SPEC),
        loop=LoopConfig(fps=30.0, exec_horizon=8),
        v_max=0.7, a_max=3.0, j_max=2500.0, filter_wn=8.0,
    )
    persist.save_config(cfg)

    loaded = AppConfig()
    assert persist.load_config_into(loaded) is True
    assert loaded.robot_node_address == "tcp://127.0.0.1:5560"
    assert loaded.cameras == [CameraConfig(role="agentview", serial="338122300664")]
    assert loaded.request_spec.connection.server_port == 9000
    assert loaded.request_spec.fields[0].name == "state"
    assert loaded.loop.fps == 30.0 and loaded.loop.exec_horizon == 8
    assert (loaded.v_max, loaded.a_max, loaded.j_max, loaded.filter_wn) == (0.7, 3.0, 2500.0, 8.0)


def test_workspace_round_trip():
    ws = WorkspaceBounds(margin=0.05)
    ws.add_point([0.2, -0.3, 0.1])
    ws.add_point([0.6, 0.3, 0.5])
    persist.save_workspace(ws)

    loaded = WorkspaceBounds()
    assert persist.load_workspace_into(loaded) is True
    assert loaded.points == [[0.2, -0.3, 0.1], [0.6, 0.3, 0.5]]
    assert loaded.margin == 0.05
    assert loaded.enabled


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
