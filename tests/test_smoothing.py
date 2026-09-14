"""EMASmoother pure-math tests -- no hardware needed.

Run: python tests/test_smoothing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from franka_deploy.safety.smoothing import EMASmoother  # noqa: E402


def test_converges_to_constant_target():
    s = EMASmoother(alpha=0.3)
    s.reset(np.zeros(7))
    target = np.full(7, 1.0)
    out = None
    for _ in range(200):
        out = s.step(target)
    assert np.allclose(out, target, atol=1e-3)


def test_step_input_is_damped_on_first_tick():
    s = EMASmoother(alpha=0.3)
    s.reset(np.zeros(7))
    out = s.step(np.full(7, 1.0))
    assert np.allclose(out, 0.3), "first tick after a step must move only alpha of the way"


def test_noisy_target_produces_smaller_output_variance():
    rng = np.random.default_rng(0)
    s = EMASmoother(alpha=0.2)
    s.reset(np.zeros(7))
    raw_history = []
    out_history = []
    for _ in range(500):
        raw = rng.uniform(-1.0, 1.0, size=7)
        raw_history.append(raw)
        out_history.append(s.step(raw))
    raw_std = np.std(raw_history, axis=0)
    out_std = np.std(out_history, axis=0)
    assert np.all(out_std < raw_std), "smoothed output must have lower variance than raw noisy input"


def test_rejects_invalid_alpha():
    for bad in [0.0, -0.1, 1.1]:
        try:
            EMASmoother(alpha=bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"alpha={bad} must be rejected"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
