"""Turns one resolved field value into whatever the request schema asks for.

This is where "송신 데이터는 형식과 구성을 직접 지정" is actually implemented:
resize/dtype/layout/normalize/encoding are all just RequestFieldSpec knobs,
so the same code path produces manipulation-stack's raw-bytes-base64 wire
format (``~/teleop-franka/manipulation-stack-dev/apps/fr3_policy_client.py``'s
``_b64``) and a user's own "PNG, then float32 [1,1,3,128,128] server-side"
protocol -- neither is special-cased.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from franka_deploy.schema.plugins import load_plugin
from franka_deploy.schema.spec import RequestFieldSpec


def _resize_hwc(img: np.ndarray, hw: list[int]) -> np.ndarray:
    import cv2

    h, w = hw
    interp = cv2.INTER_AREA if (h < img.shape[0] or w < img.shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def _apply_layout(arr: np.ndarray, layout: str) -> np.ndarray:
    if layout == "CHW" and arr.ndim == 3:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _apply_normalize(arr: np.ndarray, spec: dict[str, float]) -> np.ndarray:
    arr = arr.astype(np.float64)
    if "scale" in spec:
        arr = arr * spec["scale"]
    if "mean" in spec:
        arr = arr - spec["mean"]
    if "std" in spec:
        arr = arr / spec["std"]
    return arr


def _b64_raw(arr: np.ndarray, dtype: str) -> dict:
    arr = np.ascontiguousarray(arr, dtype=dtype)
    return {"base64": base64.b64encode(arr.tobytes()).decode(),
            "shape": list(arr.shape), "dtype": dtype}


def _encode_image_bytes(arr: np.ndarray, fmt: str) -> bytes:
    import cv2

    if arr.dtype != np.uint8:
        raise ValueError(f"{fmt} encoding requires uint8 HWC data, got dtype {arr.dtype}")
    bgr = arr[:, :, ::-1] if arr.ndim == 3 and arr.shape[2] == 3 else arr
    ok, buf = cv2.imencode(f".{fmt}", bgr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for format {fmt!r}")
    return buf.tobytes()


def _b64_image(arr: np.ndarray, fmt: str) -> dict:
    return {"base64": base64.b64encode(_encode_image_bytes(arr, fmt)).decode(), "format": fmt}


def _b64_image_str(arr: np.ndarray, fmt: str) -> str:
    """Bare base64 string, no wrapper object -- some servers (e.g. RoleVLA's
    serve_real_robot.py) expect exactly that instead of {"base64": ..., ...}."""
    return base64.b64encode(_encode_image_bytes(arr, fmt)).decode()


def build_field_value(raw_value: Any, spec: RequestFieldSpec) -> Any:
    """raw_value: whatever schema/sources.py resolved. Returns something
    ``json.dumps``-able."""
    # A bare string (a static instruction, or a custom plugin returning
    # text for a server that embeds it itself) passes straight through
    # unless the spec explicitly asks for dtype/shape handling.
    if isinstance(raw_value, str) and spec.dtype is None and spec.shape is None:
        value: Any = raw_value
    else:
        value = _build_array_value(raw_value, spec)

    if spec.transform_fn is not None:
        value = load_plugin(spec.transform_fn)(value)
    return value


def _build_array_value(raw_value: Any, spec: RequestFieldSpec) -> Any:
    arr = np.asarray(raw_value)

    if spec.resize is not None:
        arr = _resize_hwc(arr, spec.resize)

    arr = _apply_layout(arr, spec.layout)

    if spec.normalize is not None:
        arr = _apply_normalize(arr, spec.normalize)

    if spec.dtype is not None:
        arr = arr.astype(spec.dtype)

    if spec.shape is not None:
        arr = arr.reshape(spec.shape)

    if spec.encoding == "raw_bytes_base64":
        return _b64_raw(arr, spec.dtype or "uint8")
    if spec.encoding == "png_base64":
        return _b64_image(arr, "png")
    if spec.encoding == "jpeg_base64":
        return _b64_image(arr, "jpg")
    if spec.encoding == "png_base64_str":
        return _b64_image_str(arr, "png")
    # "none": plain JSON-embedded array (fine for small state/embedding vectors,
    # not for images -- use one of the base64 encodings for those).
    return arr.tolist()
