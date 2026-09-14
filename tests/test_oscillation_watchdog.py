"""OscillationWatchdog pure-math tests -- no hardware needed.

Run: python tests/test_oscillation_watchdog.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from franka_deploy.safety.watchdog import OscillationWatchdog  # noqa: E402


def test_smooth_motion_never_trips():
    wd = OscillationWatchdog(n_dims=7, window_size=20, trip_count_threshold=2)
    # Same-sign velocity the whole time -- directed motion, not vibration.
    for _ in range(200):
        tripped = wd.update(np.full(7, 0.3))
        assert not tripped


def test_alternating_sign_trips():
    wd = OscillationWatchdog(n_dims=7, window_size=20, sign_change_ratio_threshold=0.5, trip_count_threshold=2)
    tripped = False
    for i in range(200):
        dq = np.full(7, 0.5 if i % 2 == 0 else -0.5)  # flips every tick
        tripped = wd.update(dq)
        if tripped:
            break
    assert tripped, "sustained sign-alternating velocity must trip the watchdog"


def test_single_noisy_window_does_not_trip_alone():
    """One bad window shouldn't be enough -- trip_count_threshold debounces
    against a single transient."""
    wd = OscillationWatchdog(n_dims=7, window_size=20, sign_change_ratio_threshold=0.5, trip_count_threshold=5)
    for i in range(20):
        dq = np.full(7, 0.5 if i % 2 == 0 else -0.5)
        tripped = wd.update(dq)
    assert not tripped, "a single noisy window must not trip a threshold=5 watchdog"


def test_only_one_dim_oscillating_still_trips():
    wd = OscillationWatchdog(n_dims=7, window_size=20, sign_change_ratio_threshold=0.5, trip_count_threshold=2)
    tripped = False
    for i in range(200):
        dq = np.zeros(7)
        dq[4] = 0.5 if i % 2 == 0 else -0.5  # only the wrist joint chatters
        tripped = wd.update(dq)
        if tripped:
            break
    assert tripped, "oscillation in a single joint must still trip (any-dim check)"


def test_stationary_joint_sensor_noise_does_not_trip():
    """Regression test for a live false trip (2026-07-31): a joint sitting
    at ~0 still reads tiny noise straddling zero, which flips sign almost
    every tick. That must not look like oscillation just because some other
    joint is genuinely moving in the same window."""
    rng = np.random.default_rng(0)
    wd = OscillationWatchdog(n_dims=7, window_size=20, sign_change_ratio_threshold=0.5, trip_count_threshold=2)
    for _ in range(200):
        dq = rng.uniform(-0.005, 0.005, size=7)  # noise well under the 0.02 deadzone
        dq[1] = 0.3  # one joint genuinely, smoothly moving
        tripped = wd.update(dq)
        assert not tripped, "near-zero sensor noise on stationary joints must not trip the watchdog"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
