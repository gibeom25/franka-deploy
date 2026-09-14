"""GripperHysteresis pure-logic tests -- no hardware needed.

Run: python tests/test_gripper_hysteresis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from franka_deploy.robot.gripper_runtime import GripperHysteresis  # noqa: E402


def test_closes_at_threshold():
    h = GripperHysteresis(close_at=0.6, open_at=0.2, min_dwell_s=0.0)
    assert h.decide(0.3, now=0.0) is None
    assert h.decide(0.6, now=1.0) == "grasp"
    assert h.closed


def test_opens_at_threshold():
    h = GripperHysteresis(close_at=0.6, open_at=0.2, min_dwell_s=0.0)
    h.decide(0.6, now=0.0)
    assert h.closed
    assert h.decide(0.5, now=1.0) is None  # inside hysteresis band, must not reopen
    assert h.decide(0.2, now=2.0) == "move"
    assert not h.closed


def test_debounce_blocks_rapid_chatter():
    h = GripperHysteresis(close_at=0.6, open_at=0.2, min_dwell_s=0.5)
    assert h.decide(0.6, now=0.0) == "grasp"
    # Chattering back across the open threshold well within min_dwell_s must
    # be suppressed -- this is the mechanical-wear protection.
    assert h.decide(0.2, now=0.1) is None
    assert h.closed, "debounce must have blocked the premature reopen"
    # Once the dwell time has elapsed, the same crossing is honored.
    assert h.decide(0.2, now=0.6) == "move"


def test_hysteresis_band_is_stable_hold():
    h = GripperHysteresis(close_at=0.6, open_at=0.2, min_dwell_s=0.0)
    for v in [0.3, 0.4, 0.5, 0.35, 0.45]:
        assert h.decide(v, now=0.0) is None
    assert not h.closed


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
