"""Tests for schema/encode.py's field-value pipeline -- the part that makes
the outbound request format fully user-configurable (resize/dtype/layout/
normalize/encoding as independent RequestFieldSpec knobs)."""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from franka_deploy.schema.encode import build_field_value  # noqa: E402
from franka_deploy.schema.spec import RequestFieldSpec  # noqa: E402


def test_static_string_passthrough():
    spec = RequestFieldSpec(name="instruction", source="static:pick up the cup")
    assert build_field_value("pick up the cup", spec) == "pick up the cup"


def test_raw_bytes_base64_matches_manipulation_stack_b64():
    img = (np.arange(3 * 4 * 3) % 256).reshape(4, 3, 3).astype(np.uint8)
    spec = RequestFieldSpec(name="obs", source="camera:agentview", dtype="uint8", encoding="raw_bytes_base64")
    out = build_field_value(img, spec)
    assert out["shape"] == [4, 3, 3]
    assert out["dtype"] == "uint8"
    decoded = np.frombuffer(base64.b64decode(out["base64"]), dtype=np.uint8).reshape(4, 3, 3)
    assert np.array_equal(decoded, img)


def test_resize_layout_dtype_shape_pipeline():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    spec = RequestFieldSpec(
        name="agentview_image", source="camera:agentview", dtype="float32",
        shape=[1, 1, 3, 32, 32], resize=[32, 32], layout="CHW",
        normalize={"scale": 1.0 / 255.0},
    )
    out = build_field_value(img, spec)
    # encoding="none" JSON-embeds a plain nested list -- JSON has no dtype
    # of its own, so `dtype` here only controls precision *before* the
    # float64-by-default json round trip, not what comes back out of it.
    arr = np.asarray(out, dtype=np.float32)
    assert arr.shape == (1, 1, 3, 32, 32)
    assert np.allclose(arr[0, 0, 0], 1.0)  # red channel, scaled to 1.0
    assert np.allclose(arr[0, 0, 1], 0.0)


def test_png_base64_round_trips():
    import cv2

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    spec = RequestFieldSpec(name="img", source="camera:agentview", encoding="png_base64")
    out = build_field_value(img, spec)
    assert out["format"] == "png"
    buf = base64.b64decode(out["base64"])
    decoded_bgr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = decoded_bgr[:, :, ::-1]
    assert np.allclose(decoded_rgb, img, atol=2)  # PNG is lossless; small tolerance for color conv


def test_custom_transform_fn_runs_after_shaping(tmp_path, monkeypatch):
    plugin_file = tmp_path / "my_plugins.py"
    plugin_file.write_text("def double(value):\n    return [v * 2 for v in value]\n")
    monkeypatch.setenv("FRANKA_DEPLOY_PLUGIN_DIR", str(tmp_path))
    import franka_deploy.schema.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "PLUGIN_DIR", tmp_path)
    plugins_mod.clear_cache()

    spec = RequestFieldSpec(name="state", source="robot:joint_positions",
                             dtype="float32", transform_fn="my_plugins.double")
    out = build_field_value(np.array([1.0, 2.0, 3.0]), spec)
    assert out == [2.0, 4.0, 6.0]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
