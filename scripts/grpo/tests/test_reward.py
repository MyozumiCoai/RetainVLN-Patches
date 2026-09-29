import math
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from scripts.grpo.reward import (
    FWD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    in_contract,
    normalize_group,
    parse_action,
    reward,
)


def test_parse_forward_wrapped():
    assert parse_action("The next action is move forward 75 cm.") == (FWD, 75.0)


def test_parse_forward_plain():
    assert parse_action("move forward 25 cm") == (FWD, 25.0)


def test_parse_forward_meter_unit():
    a = parse_action("move forward 0.75 m")
    assert a is not None and a[0] == FWD and abs(a[1] - 75.0) < 1e-6


def test_parse_stop():
    assert parse_action("The next action is stop.") == (STOP, None)
    assert parse_action("stop") == (STOP, None)


def test_parse_turn_left_degree_symbol():
    assert parse_action("turn left 30 degree") == (TURN_LEFT, 30.0)
    assert parse_action("turn right 45°") == (TURN_RIGHT, 45.0)


def test_parse_garbage_returns_none():
    assert parse_action("hello world") is None
    assert parse_action("") is None
    assert parse_action(None) is None
    assert parse_action("move forward") is None


def test_parse_turn_priority_over_stop_word():
    assert parse_action("turn left 15 degree") == (TURN_LEFT, 15.0)


def test_in_contract_grid():
    assert in_contract((FWD, 75.0))
    assert in_contract((FWD, 25.0))
    assert not in_contract((FWD, 60.0))
    assert in_contract((TURN_LEFT, 30.0))
    assert not in_contract((TURN_LEFT, 20.0))
    assert in_contract((STOP, None))
    assert not in_contract(None)
    assert not in_contract((FWD, None))


def test_reward_exact_forward_match():
    r = reward("move forward 75 cm", "move forward 75 cm")
    assert abs(r - (1.0 + 0.5)) < 1e-9


def test_reward_forward_type_match_param_off():
    r = reward("move forward 50 cm", "move forward 75 cm")
    expect = 1.0 + 0.5 * math.exp(-25.0 / 25.0)
    assert abs(r - expect) < 1e-9


def test_reward_type_mismatch_in_contract_is_zero():
    r = reward("turn left 30 degree", "move forward 75 cm")
    assert abs(r - 0.0) < 1e-9


def test_reward_stop_correct():
    r = reward("stop", "stop")
    assert abs(r - (1.0 + 1.0)) < 1e-9


def test_reward_stop_wrong_when_gt_forward():
    r = reward("stop", "move forward 75 cm")
    assert abs(r - 0.0) < 1e-9


def test_reward_parse_fail_penalized():
    r = reward("nonsense output", "stop")
    assert abs(r - (-0.5)) < 1e-9


def test_reward_out_of_contract_but_type_match():
    r = reward("move forward 60 cm", "move forward 75 cm")
    expect = -0.5 + 1.0 + 0.5 * math.exp(-15.0 / 25.0)
    assert abs(r - expect) < 1e-9


def test_reward_gt_unparsable_raises():
    try:
        reward("stop", "do a backflip")
    except ValueError:
        return
    raise AssertionError("GT 不可解析时应抛 ValueError")


def test_reward_wrapped_pred_matches():
    assert abs(
        reward("The next action is turn right 45 degree.", "turn right 45 degree")
        - (1.0 + 0.5)
    ) < 1e-9


def test_normalize_group_zero_variance_degenerate():
    adv, degen = normalize_group([1.0, 1.0, 1.0])
    assert degen is True
    assert adv == [0.0, 0.0, 0.0]


def test_normalize_group_basic():
    adv, degen = normalize_group([0.0, 1.0, 2.0])
    assert degen is False
    assert abs(sum(adv) / len(adv)) < 1e-6
    assert abs(adv[0] - (-1.2247448713915892)) < 1e-6
    assert adv[2] > adv[1] > adv[0]


def test_normalize_group_empty():
    adv, degen = normalize_group([])
    assert adv == [] and degen is True


def test_v2_geo_missing_is_v1():
    a = reward("move forward 75 cm", "move forward 75 cm")
    b = reward("move forward 75 cm", "move forward 75 cm", geo_t=1.5, geo_after=None)
    c = reward("move forward 75 cm", "move forward 75 cm", geo_t=None, geo_after=1.2)
    assert a == b == c == 1.0 + 0.5


def test_v2_forward_full_progress():
    r = reward("move forward 25 cm", "move forward 25 cm", geo_t=2.0, geo_after=1.75)
    assert abs(r - (1.0 + 0.5 + 1.0)) < 1e-6


def test_v2_forward_partial_progress():
    r = reward("move forward 25 cm", "move forward 75 cm", geo_t=3.0, geo_after=2.25)
    base = 1.0 + 0.5 * math.exp(-50.0 / 25.0)
    assert abs(r - (base + 1.0 / 3.0)) < 1e-6


def test_v2_forward_overshoot_clipped():
    r = reward("move forward 75 cm", "move forward 25 cm", geo_t=2.0, geo_after=1.75)
    base = 1.0 + 0.5 * math.exp(-50.0 / 25.0)
    assert abs(r - (base + 1.0)) < 1e-6


def test_v2_turn_no_progress():
    r = reward("turn left 30 degree", "turn left 30 degree", geo_t=4.0, geo_after=4.0)
    assert abs(r - (1.0 + 0.5)) < 1e-6


def test_v2_stop_ramp_at_goal():
    r = reward("stop", "move forward 25 cm", geo_t=0.0, geo_after=-0.25)
    assert abs(r - 1.0) < 1e-6


def test_v2_stop_ramp_partial_ok():
    r = reward("stop", "move forward 25 cm", geo_t=0.5, geo_after=0.25)
    assert abs(r - (1.0 - 0.5 / 1.5)) < 1e-6


def test_v2_stop_ramp_zero_at_radius_ok():
    r = reward("stop", "move forward 25 cm", geo_t=1.5, geo_after=1.25)
    assert abs(r) < 1e-6


def test_v2_stop_soft_penalty_mid():
    r = reward("stop", "move forward 25 cm", geo_t=2.0, geo_after=1.75)
    assert abs(r - (-0.5 * 0.5 / 1.5)) < 1e-6


def test_v2_stop_early_hard_penalty():
    r = reward("stop", "move forward 25 cm", geo_t=3.0, geo_after=2.75)
    assert abs(r - (-1.0)) < 1e-6
    r = reward("stop", "move forward 25 cm", geo_t=8.0, geo_after=7.75)
    assert abs(r - (-1.0)) < 1e-6


def test_v2_stop_correct_gt_stack():
    r = reward("stop", "stop", geo_t=0.3, geo_after=None)
    assert abs(r - (1.0 + 1.0 + 0.8)) < 1e-6


def test_v2_stop_gt_but_far():
    r = reward("stop", "stop", geo_t=9.0, geo_after=None)
    assert abs(r - (1.0 + 1.0 - 1.0)) < 1e-6


def test_v2_stop_ramp_monotonic_decreasing():
    vals = [reward("stop", "move forward 25 cm", geo_t=g, geo_after=g - 0.25) for g in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)), vals


def test_v2_forward_wrong_type_no_progress():
    r = reward("move forward 25 cm", "turn left 30 degree", geo_t=5.0, geo_after=4.0)
    assert r == 0.0
def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        fn()
        passed += 1
        print(f"  PASS {name}")
    print(f"\n[test_reward] {passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
