import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
import numpy as np
from navila_patches.obs_strategy import sample_and_pad_images, _cfg
from navila_patches.cot_prompt import build_question, parse_cot_output


def make_pool(n: int, pattern: str = "vary") -> list:
    imgs = []
    for i in range(n):
        if pattern == "flat":
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
        elif pattern == "duplicate":
            arr = np.full((128, 128, 3), 128, dtype=np.uint8)
        else:
            arr = np.full((128, 128, 3), (i * 10) % 256, dtype=np.uint8)
        imgs.append(Image.fromarray(arr))
    return imgs


def assert_eq(a, b, msg=""):
    assert a == b, f"FAIL {msg}: {a!r} != {b!r}"


def assert_in_range(x, lo, hi, msg=""):
    assert lo <= x <= hi, f"FAIL {msg}: {x} not in [{lo},{hi}]"


def test_linspace_equivalence():
    import copy as _copy

    def original_navila(images, num_frames=8, width=512, height=512):
        frames = _copy.deepcopy(images)
        if len(frames) < num_frames:
            while len(frames) < num_frames:
                frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
        latest_frame = frames[-1]
        sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
        return [frames[i] for i in sampled_indices] + [latest_frame]

    os.environ["NAVI_OBS_MODE"] = "linspace"
    os.environ["NAVI_OBS_GAMMA"] = "0.0"
    os.environ["NAVI_OBS_BUDGET"] = "8"
    for L in (1, 3, 8, 9, 10, 12, 20, 50):
        pool = make_pool(L, "vary")
        out = sample_and_pad_images(pool, num_frames=8)
        ref = original_navila(pool, num_frames=8)
        assert_eq(len(out), 8, f"L={L} len")
        assert_eq(len(out), len(ref), f"L={L} 长度一致")
        for j, (a, b) in enumerate(zip(out, ref)):
            assert np.array_equal(np.array(a), np.array(b)), f"L={L} 第 {j} 帧像素不一致"
    print("  test_linspace_equivalence PASS")


def test_padding_when_too_few():
    os.environ["NAVI_OBS_MODE"] = "linspace"
    os.environ["NAVI_OBS_BUDGET"] = "8"
    pool = make_pool(3, "vary")
    out = sample_and_pad_images(pool, num_frames=8)
    assert_eq(len(out), 8, "padding len")
    for i in range(5):
        arr = np.asarray(out[i])
        assert_eq(arr.sum(), 0, f"前 {i} 帧应为黑")
    assert np.array_equal(np.array(out[-1]), np.array(pool[-1])), "padding latest 保留"
    print("  test_padding_when_too_few PASS")


def test_recency_bias():
    from navila_patches.obs_strategy import _recency_indices
    rng0 = np.random.default_rng(42)
    rng3 = np.random.default_rng(42)
    n = 50; k = 7
    out0 = _recency_indices(n, k, gamma=0.0, rng=rng0)
    out3 = _recency_indices(n, k, gamma=3.0, rng=rng3)
    a0, a3 = out0.mean(), out3.mean()
    print(f"    recency: gamma=0 avg={a0:.1f}, gamma=3 avg={a3:.1f}")
    assert a3 > a0, f"gamma=3 应比 gamma=0 选得更靠后，但 {a3} <= {a0}"
    assert_eq(list(out0), sorted(out0), "recency gamma=0 时序")
    assert_eq(list(out3), sorted(out3), "recency gamma=3 时序")
    print("  test_recency_bias PASS")


def test_random_uniform():
    os.environ["NAVI_OBS_MODE"] = "random_uniform"
    os.environ["NAVI_OBS_SEED"] = "0"
    pool = make_pool(20, "vary")
    out = sample_and_pad_images(pool, num_frames=8)
    assert_eq(len(out), 8, "random len")
    assert np.array_equal(np.array(out[-1]), np.array(pool[-1])), "random latest 保留"
    selected = out[:-1]
    arrs = [np.array(f) for f in selected]
    unique = set(tuple(a.flatten()[::100].tolist()) for a in arrs)
    assert_eq(len(unique), 7, "random 不放回（内容两两不同）")
    print("  test_random_uniform PASS")


def test_three_criteria_prefers_diverse():
    from navila_patches.obs_strategy import _three_criteria_indices
    pool = make_pool(20, "vary")
    idx = _three_criteria_indices(pool, k=5, rng=np.random.default_rng(0))
    assert_eq(len(idx), 5, "tc 长度")
    assert_eq(len(set(idx.tolist())), 5, "tc 不放回")
    assert_eq(idx.tolist(), sorted(idx.tolist()), "tc 时序单调")
    print(f"    three_criteria 选中的 idx: {idx.tolist()}")
    print("  test_three_criteria_prefers_diverse PASS")


def test_cot_prompt_build():
    os.environ["NAVI_COT"] = "0"
    q0 = build_question("go to kitchen", "<image>\n" * 7)
    assert "go to kitchen" in q0
    assert "<think>" not in q0
    print("    cot off: prompt 无 CoT 标签指令 OK")

    os.environ["NAVI_COT"] = "1"
    os.environ["NAVI_COT_TEMPLATE"] = "minimal"
    q1 = build_question("go to kitchen", "<image>\n" * 7)
    assert "<think>" in q1 and "<action>" in q1
    assert "go to kitchen" in q1
    print("    cot on (minimal): 含 <think>/<action> 指令 OK")

    os.environ["NAVI_COT_TEMPLATE"] = "detailed"
    q2 = build_question("go to kitchen", "<image>\n" * 7)
    assert "Scene:" in q2 and "Progress:" in q2
    print("    cot on (detailed): 含 Scene/Progress/Decision 三段 OK")
    print("  test_cot_prompt_build PASS")


def test_cot_parser_standard():
    raw = "<think>I see a hallway ahead, no kitchen visible. I should move forward.</think><action>move forward 25 cm</action>"
    action, extras = parse_cot_output(raw)
    assert_eq(action, 1, "标准 CoT -> move_forward")
    assert_eq(extras["distance"], 25, "distance 提取")
    assert_in_range(len(extras["think"] or ""), 5, 200, "think 提取")
    assert "move forward" in (extras["action_text"] or "")
    print("  test_cot_parser_standard PASS")


def test_cot_parser_stop_and_turn():
    cases = [
        ("<action>stop</action>",                          0, None,  None),
        ("I think we arrived. <action>stop</action>",       0, None,  None),
        ("<action>turn left 30 degree</action>",            2, None,  30),
        ("<action>turn right 45 degree</action>",           3, None,  45),
    ]
    for raw, exp_action, _, exp_degree in cases:
        action, extras = parse_cot_output(raw)
        assert_eq(action, exp_action, f"action for {raw[:30]}")
        if exp_degree is not None:
            assert_eq(extras["degree"], exp_degree, f"degree for {raw[:30]}")
    print("  test_cot_parser_stop_and_turn PASS")


def test_cot_parser_fallback_no_tags():
    raw = "Based on the scene, the next action is move forward 50 cm to get closer."
    action, extras = parse_cot_output(raw)
    assert_eq(action, 1, "无标签回退 -> move_forward")
    assert_eq(extras["distance"], 50, "无标签 distance 提取")
    print("  test_cot_parser_fallback_no_tags PASS")


def test_cot_parser_unparseable_defaults_to_forward():
    raw = "I have no idea what to do here."
    action, extras = parse_cot_output(raw)
    assert action is None, f"无法识别应返回 None（调用方 _fallback_action 才接管），但得到 {action}"
    print("  test_cot_parser_unparseable_defaults_to_forward PASS")


def main():
    print("=" * 60)
    print("navila_patches offline smoke test")
    print("=" * 60)
    tests = [
        test_linspace_equivalence,
        test_padding_when_too_few,
        test_recency_bias,
        test_random_uniform,
        test_three_criteria_prefers_diverse,
        test_cot_prompt_build,
        test_cot_parser_standard,
        test_cot_parser_stop_and_turn,
        test_cot_parser_fallback_no_tags,
        test_cot_parser_unparseable_defaults_to_forward,
    ]
    failed = 0
    for t in tests:
        try:
            print(f"\n[{t.__name__}]")
            t()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1
    print("\n" + "=" * 60)
    print(f"result: {len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
