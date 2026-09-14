"""Pure-math tests for JerkLimitedReferenceFilter -- no hardware needed.

Run with pytest if available, or directly:
    python tests/test_reference_filter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from franka_deploy.robot.reference_filter import JerkLimitedReferenceFilter  # noqa: E402


def _run(filt: JerkLimitedReferenceFilter, target: np.ndarray, n_steps: int):
    """Step the filter n_steps times toward a fixed target; return the
    per-step qd and acc history for limit checking."""
    qd_hist = [filt.qd_cmd.copy()]
    acc_hist = [filt._acc_prev.copy()]
    for _ in range(n_steps):
        filt.step(target)
        qd_hist.append(filt.qd_cmd.copy())
        acc_hist.append(filt._acc_prev.copy())
    return np.array(qd_hist), np.array(acc_hist)


def test_converges_to_target():
    filt = JerkLimitedReferenceFilter(n_dims=7, dt=1e-3, v_max=1.0, a_max=4.0, j_max=3000.0, wn=10.0)
    q0 = np.zeros(7)
    target = np.full(7, 0.5)
    filt.reset(q0)
    for _ in range(5000):  # 5 s of sim time, plenty for wn=10
        q_cmd = filt.step(target)
    assert np.allclose(q_cmd, target, atol=1e-3), f"did not converge: {q_cmd}"


def test_velocity_never_exceeds_v_max():
    filt = JerkLimitedReferenceFilter(n_dims=7, dt=1e-3, v_max=1.0, a_max=4.0, j_max=3000.0, wn=10.0)
    filt.reset(np.zeros(7))
    # A large step target stresses the saturation the hardest.
    target = np.full(7, 10.0)
    qd_hist, _ = _run(filt, target, 3000)
    assert np.all(np.abs(qd_hist) <= 1.0 + 1e-9), f"v_max violated: max={np.abs(qd_hist).max()}"


def test_acceleration_never_exceeds_a_max():
    filt = JerkLimitedReferenceFilter(n_dims=7, dt=1e-3, v_max=1.0, a_max=4.0, j_max=3000.0, wn=10.0)
    filt.reset(np.zeros(7))
    target = np.full(7, 10.0)
    _, acc_hist = _run(filt, target, 3000)
    assert np.all(np.abs(acc_hist) <= 4.0 + 1e-9), f"a_max violated: max={np.abs(acc_hist).max()}"


def test_jerk_never_exceeds_j_max():
    filt = JerkLimitedReferenceFilter(n_dims=7, dt=1e-3, v_max=1.0, a_max=4.0, j_max=3000.0, wn=10.0)
    filt.reset(np.zeros(7))
    target = np.full(7, 10.0)
    _, acc_hist = _run(filt, target, 3000)
    jerk = np.diff(acc_hist, axis=0) / filt.dt
    assert np.all(np.abs(jerk) <= 3000.0 + 1e-6), f"j_max violated: max={np.abs(jerk).max()}"


def test_discontinuous_target_does_not_spike_command():
    """A noisy/jumpy target (like a raw policy output) must not translate
    into a q_cmd jump -- this is the whole reason the filter exists."""
    filt = JerkLimitedReferenceFilter(n_dims=7, dt=1e-3, v_max=1.0, a_max=4.0, j_max=3000.0, wn=10.0)
    q0 = np.zeros(7)
    filt.reset(q0)
    rng = np.random.default_rng(0)
    prev_q_cmd = q0.copy()
    for _ in range(2000):
        noisy_target = rng.uniform(-1.0, 1.0, size=7)  # discontinuous every tick
        q_cmd = filt.step(noisy_target)
        step = np.abs(q_cmd - prev_q_cmd).max()
        assert step <= 1.0 * filt.dt + 1e-9, f"q_cmd jumped {step} in one tick"
        prev_q_cmd = q_cmd


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
