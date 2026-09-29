import math
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from scripts.grpo.memory_inject import (
    build_step_memory,
    inject,
    memory_for_step,
    poses_from_rows,
)


def _straight(n=20, dx=0.5, heading=0.0):
    return [((i * dx, 0.0, 0.0), heading) for i in range(n)]


def test_length_and_nonempty():
    poses = _straight(12)
    texts = build_step_memory(poses)
    assert len(texts) == len(poses)
    assert all(isinstance(t, str) and t for t in texts)


def test_memory_for_step_equals_cumulative():
    poses = _straight(15)
    full = build_step_memory(poses)
    for k in (0, 1, 5, 9, 14):
        assert memory_for_step(poses, k) == full[k]


def test_memory_for_step_prefix_only():
    poses = _straight(10)
    before = memory_for_step(poses, 4)
    poses2 = list(poses[:5]) + [((99.0, 0.0, 99.0), math.pi) for _ in range(5)]
    assert memory_for_step(poses2, 4) == before


def test_memory_for_step_out_of_range():
    poses = _straight(5)
    for bad in (-1, 5, 99):
        try:
            memory_for_step(poses, bad)
        except IndexError:
            continue
        raise AssertionError(f"step={bad} 应抛 IndexError")


def test_progress_appears():
    texts = build_step_memory(_straight(12, dx=0.5))
    assert "已移动" in texts[-1] and "步" in texts[-1]


def test_new_region_appears():
    texts = build_step_memory(_straight(20, dx=0.5))
    assert any("进入新区域" in t for t in texts)


def test_turn_event_appears():
    poses = [((0.0, 0.0, 0.0), 0.0), ((0.0, 0.0, 0.0), 0.0)]
    poses.append(((0.0, 0.0, 0.0), math.pi / 2))
    texts = build_step_memory(poses)
    assert any("转角" in t and "左转" in t for t in texts)


def test_loop_closure_appears():
    poses = [((i * 1.0, 0.0, 0.0), 0.0) for i in range(40)]
    poses.append(((0.5, 0.0, 0.0), 0.0))
    texts = build_step_memory(poses)
    assert any("回到已访问区域" in t for t in texts)


def test_level1_no_notebook_level2_has():
    poses = _straight(20, dx=0.5)
    l1 = build_step_memory(poses, level="level1")[-1]
    l2 = build_step_memory(poses, level="level2")[-1]
    assert "Notebook" not in l1 and "Searched" not in l1
    assert "Searched" in l2 or "Notebook" in l2


def test_bad_level_raises():
    try:
        build_step_memory(_straight(3), level="level9")
    except ValueError:
        return
    raise AssertionError("非法 level 应抛 ValueError")


def test_no_oracle_leak():
    poses = _straight(20, dx=0.5)
    texts = build_step_memory(poses)
    joined = "\n".join(texts).lower()
    for banned in ("geodesic", "goal", "目标", "oracle", "50.0", "success", "成功到达"):
        assert banned not in joined, f"记忆文本泄漏了 {banned!r}"


def test_outcome_stays_in_progress():
    texts = build_step_memory(_straight(10))
    assert all("进行中" in t for t in texts)
    assert not any("成功到达" in t or "未成功" in t for t in texts)


def test_inject_prepends():
    assert inject("QUESTION", "MEM") == "MEM\nQUESTION"


def test_inject_empty_passthrough():
    assert inject("QUESTION", "") == "QUESTION"
    assert inject("QUESTION", None) == "QUESTION"


def test_poses_from_rows_primary_keys():
    rows = [
        {"position": [0.0, 0.0, 0.0], "heading_rad": 0.0},
        {"position": [0.5, 0.0, 0.0], "heading_rad": 0.1},
    ]
    poses = poses_from_rows(rows)
    assert len(poses) == 2
    assert poses[1][0] == [0.5, 0.0, 0.0] and abs(poses[1][1] - 0.1) < 1e-9


def test_poses_from_rows_alt_keys():
    rows = [{"current_position": [1.0, 0.0, 2.0], "heading": 0.5}]
    poses = poses_from_rows(rows)
    assert poses[0][0] == [1.0, 0.0, 2.0] and abs(poses[0][1] - 0.5) < 1e-9


def test_poses_from_rows_missing_raises():
    try:
        poses_from_rows([{"position": [0.0, 0.0, 0.0]}])
    except KeyError:
        return
    raise AssertionError("缺字段应抛 KeyError")


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        fn()
        passed += 1
        print(f"  PASS {name}")
    print(f"\n[test_memory_inject] {passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
