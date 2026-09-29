import json
import os
import sys
import tempfile

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from scripts.grpo.dataset import attach_memory, group_by_episode, prepare_samples
from scripts.grpo.memory_inject import build_step_memory, memory_for_step


def _pt(scene, ep, t, action="move forward 25 cm"):
    return {
        "id": f"{scene}_ep{ep}_t{t}",
        "scene_id": scene,
        "episode_id": ep,
        "timestep": t,
        "ep_len": 3,
        "instruction": "walk",
        "images": [f"F:/MP3D/{scene}/f{t}.jpg"],
        "gt_action": action,
        "action_class": "forward" if action.startswith("move") else "turn",
        "position": [float(t), 0.0, 0.0],
        "heading_rad": 0.3 * t,
        "direct_answer": f"The next action is {action}.",
        "geodesic_distance": 5.0 - t,
        "next_position": [t + 1.0, 0.0, 0.0],
        "goal_position": [9.0, 0.0, 9.0],
    }


def synth_points():
    pts = []
    for t, a in enumerate(["move forward 25 cm", "turn left 30 degree", "move forward 50 cm"]):
        pts.append(_pt("S1", 1, t, a))
    for t, a in enumerate(["turn right 45 degree", "move forward 75 cm"]):
        pts.append(_pt("S2", 6, t, a))
    return pts


def test_group_by_episode_sorted():
    shuffled = [synth_points()[3], synth_points()[0], synth_points()[4], synth_points()[1], synth_points()[2]]
    g = group_by_episode(shuffled)
    assert set(g.keys()) == {("S1", 1), ("S2", 6)}
    assert [r["timestep"] for r in g[("S1", 1)]] == [0, 1, 2]


def test_attach_memory_level1_nonempty_and_count():
    out = attach_memory(synth_points(), memory_level="level1")
    assert len(out) == 5
    assert all(s["memory_text"] for s in out)
    assert all(s["memory_level"] == "level1" for s in out)


def test_attach_memory_matches_memory_for_step():
    pts = synth_points()
    out = attach_memory(pts, memory_level="level1")
    ep1 = [p for p in pts if p["scene_id"] == "S1"]
    poses = [(p["position"], p["heading_rad"]) for p in sorted(ep1, key=lambda x: x["timestep"])]
    by_t = {s["timestep"]: s for s in out if s["scene_id"] == "S1"}
    for t in (0, 1, 2):
        assert by_t[t]["memory_text"] == memory_for_step(poses, t)


def test_attach_memory_off_is_empty():
    out = attach_memory(synth_points(), memory_level="off")
    assert all(s["memory_text"] == "" for s in out)
    assert all(s["memory_level"] == "off" for s in out)


def test_attach_memory_level2_has_notebook():
    pts = [_pt("S1", 1, t) for t in range(1)]
    long_pts = []
    for t in range(12):
        p = _pt("S1", 9, t)
        p["position"] = [t * 1.0, 0.0, 0.0]
        long_pts.append(p)
    out = attach_memory(long_pts, memory_level="level2")
    assert "Searched" in out[-1]["memory_text"] or "Notebook" in out[-1]["memory_text"]
    assert pts


def test_attach_memory_gapped_raises():
    gapped = [_pt("S1", 1, 0), _pt("S1", 1, 2)]
    try:
        attach_memory(gapped, memory_level="level1")
    except ValueError:
        return
    raise AssertionError("缺口 episode 应抛 ValueError")


def test_attach_memory_bad_level_raises():
    try:
        attach_memory(synth_points(), memory_level="levelX")
    except ValueError:
        return
    raise AssertionError("非法 level 应抛 ValueError")


def test_prepare_samples_from_file():
    pts = synth_points()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "grpo_train.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in pts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        out = prepare_samples(path, memory_level="level1")
    assert len(out) == 5
    assert all("memory_text" in s and s["memory_text"] for s in out)
    assert {s["gt_action"] for s in out if s["scene_id"] == "S1"} == {
        "move forward 25 cm", "turn left 30 degree", "move forward 50 cm"
    }


def test_prepare_samples_off_file():
    pts = synth_points()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in pts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        out = prepare_samples(path, memory_level="off")
    assert all(s["memory_text"] == "" for s in out)



def test_v2_prefix_poses_noncontiguous_rows():
    from scripts.grpo.dataset import attach_memory

    pts = [
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 10145, "timestep": 15,
         "position": [1.0, 0.0, 1.0], "heading_rad": 0.0, "images": ["f"],
         "gt_action": "move forward 25 cm",
         "prefix_poses": [[float(i), 0.0, float(i), 0.0] for i in range(16)]},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 10145, "timestep": 22,
         "position": [1.0, 0.0, 1.0], "heading_rad": 0.0, "images": ["f"],
         "gt_action": "stop",
         "prefix_poses": [[float(i), 0.0, float(i), 0.0] for i in range(23)]},
    ]
    out = attach_memory(pts, memory_level="level1", budget=400)
    assert len(out) == 2
    for r in out:
        assert isinstance(r["memory_text"], str) and len(r["memory_text"]) > 0
    t22 = next(r for r in out if r["timestep"] == 22)
    t15 = next(r for r in out if r["timestep"] == 15)
    assert t22["memory_text"] != t15["memory_text"]


def test_v2_prefix_poses_off():
    from scripts.grpo.dataset import attach_memory

    pts = [{"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 3,
            "prefix_poses": [[0.0, 0.0, 0.0, 0.0]] * 4}]
    out = attach_memory(pts, memory_level="off")
    assert out[0]["memory_text"] == ""


def test_v1_contiguity_still_enforced():
    from scripts.grpo.dataset import attach_memory

    pts = [
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 0,
         "position": [0.0, 0.0, 0.0], "heading_rad": 0.0, "images": ["a"],
         "gt_action": "move forward 25 cm"},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 2,
         "position": [0.0, 0.0, 0.0], "heading_rad": 0.0, "images": ["b"],
         "gt_action": "move forward 25 cm"},
    ]
    try:
        attach_memory(pts, memory_level="level1")
        raised = False
    except ValueError:
        raised = True
    assert raised

def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        fn()
        passed += 1
        print(f"  PASS {name}")
    print(f"\n[test_dataset] {passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()


def test_v2_prefix_poses_noncontiguous_rows():
    from scripts.grpo.dataset import attach_memory

    pts = [
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 10145, "timestep": 15,
         "position": [1.0, 0.0, 1.0], "heading_rad": 0.0, "images": ["f"],
         "gt_action": "move forward 25 cm",
         "prefix_poses": [[float(i), 0.0, float(i), 0.0] for i in range(16)]},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 10145, "timestep": 22,
         "position": [1.0, 0.0, 1.0], "heading_rad": 0.0, "images": ["f"],
         "gt_action": "stop",
         "prefix_poses": [[float(i), 0.0, float(i), 0.0] for i in range(23)]},
    ]
    out = attach_memory(pts, memory_level="level1", budget=400)
    assert len(out) == 2
    for r in out:
        assert isinstance(r["memory_text"], str) and len(r["memory_text"]) > 0
    t22 = next(r for r in out if r["timestep"] == 22)
    t15 = next(r for r in out if r["timestep"] == 15)
    assert t22["memory_text"] != t15["memory_text"]


def test_v2_prefix_poses_off():
    from scripts.grpo.dataset import attach_memory

    pts = [{"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 3,
            "prefix_poses": [[0.0, 0.0, 0.0, 0.0]] * 4}]
    out = attach_memory(pts, memory_level="off")
    assert out[0]["memory_text"] == ""


def test_v1_contiguity_still_enforced():
    from scripts.grpo.dataset import attach_memory

    pts = [
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 0,
         "position": [0.0, 0.0, 0.0], "heading_rad": 0.0, "images": ["a"],
         "gt_action": "move forward 25 cm"},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 2,
         "position": [0.0, 0.0, 0.0], "heading_rad": 0.0, "images": ["b"],
         "gt_action": "move forward 25 cm"},
    ]
    try:
        attach_memory(pts, memory_level="level1")
        raised = False
    except ValueError:
        raised = True
    assert raised
