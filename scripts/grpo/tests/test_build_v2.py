import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from scripts.grpo.build_grpo_v2 import (
    EVAL_SCENES,
    EXCLUDED_SOURCES,
    action_class,
    attach_geo_after,
    build,
    densify_gt,
    in_contract,
    last_step_index,
    norm_scene,
    r2r_rows_to_clean,
    terminal_row_to_point,
)

GT_EP = {
    "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]],
    "goals": [{"position": [1.0, 0.0, 3.0]}],
}


def _gt_ep_for(scene, ep):
    e = json.loads(json.dumps(GT_EP))
    return e


def _row(action, t, images=None, scene="ur6pFq6Qu1A", ep=1, source="Terminal-v1", **kw):
    r = {
        "id": f"t3_{scene}_ep{ep}_t{t:03d}",
        "source": source,
        "scene_id": scene,
        "episode_id": ep,
        "timestep": t,
        "instruction": "go to the table",
        "images": images or [f"/s/frames/ep{ep}/step{t:04d}.png"],
        "action": action,
        "geodesic_to_goal": 1.5,
    }
    r.update(kw)
    return r


def test_norm_scene():
    assert norm_scene("mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.glb") == "ur6pFq6Qu1A"
    assert norm_scene("ur6pFq6Qu1A") == "ur6pFq6Qu1A"


def test_densify_gt_geometry():
    subs = densify_gt(_gt_ep_for("ur6pFq6Qu1A", 1))
    assert len(subs) == 16, len(subs)
    gx, gz = subs[-1]["position"][0], subs[-1]["position"][2]
    assert abs(gx - 1.0) < 1e-6 and abs(gz - 3.0) < 1e-6
    assert abs(subs[0]["heading_rad"]) < 1e-6
    assert abs(subs[3]["heading_rad"] - 0.0) < 1e-6
    assert abs(subs[4]["heading_rad"] - math.pi / 2) < 1e-6


def test_terminal_row_aligned():
    subs = densify_gt(_gt_ep_for("ur6pFq6Qu1A", 1))
    r = _row("move forward 75 cm", t=4, images=[f"/s/frames/ep1/step{i:04d}.png" for i in range(4 - 7, 4 + 1)])
    p = terminal_row_to_point(r, GT_EP, subs)
    assert p is not None
    assert p["position"] == subs[4]["position"]
    assert p["heading_rad"] == subs[4]["heading_rad"]
    assert p["gt_action"] == "move forward 75 cm"
    assert p["next_position"] == subs[5]["position"]
    assert p["goal_position"] == [1.0, 0.0, 3.0]
    assert p["ep_len"] == 16


def test_terminal_row_stop_keeps_pose():
    subs = densify_gt(_gt_ep_for("ur6pFq6Qu1A", 1))
    r = _row("stop", t=11, images=[f"/s/frames/ep1/step{i:04d}.png" for i in range(4, 12)])
    p = terminal_row_to_point(r, GT_EP, subs)
    assert p is not None
    assert p["action_class"] == "stop"
    assert p["position"] == subs[11]["position"]


def test_terminal_row_misaligned_frame_filtered():
    subs = densify_gt(_gt_ep_for("ur6pFq6Qu1A", 1))
    r = _row("move forward 25 cm", t=3, images=["/s/frames/ep1/step0099.png"])
    assert terminal_row_to_point(r, GT_EP, subs) is None


def test_terminal_row_out_of_range_filtered():
    subs = densify_gt(_gt_ep_for("ur6pFq6Qu1A", 1))
    r = _row("move forward 25 cm", t=99)
    assert terminal_row_to_point(r, GT_EP, subs) is None


def test_terminal_out_of_contract_filtered():
    subs = densify_gt(_gt_ep_for("ur6pFq6Qu1A", 1))
    r = _row("turn left 10 degree", t=1)
    assert terminal_row_to_point(r, GT_EP, subs) is None


def test_r2r_stop_inherits_prev_pose():
    rows = [
        {"timestep": 0, "action": "move forward 25 cm", "current_position": [0.0, 0.0, 0.0],
         "current_heading_rad": 0.5, "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 5,
         "images": ["a.jpg"], "instruction": "x", "id": "r1"},
        {"timestep": 1, "action": "stop", "current_position": [0.25, 0.0, 0.0],
         "current_heading_rad": None, "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 5,
         "images": ["b.jpg"], "instruction": "x", "id": "r2"},
    ]
    clean = r2r_rows_to_clean(rows)
    assert (norm_scene("ur6pFq6Qu1A"), 5) in clean
    stop_row = clean[(norm_scene("ur6pFq6Qu1A"), 5)][1]
    assert stop_row["current_heading_rad"] == 0.5
    assert stop_row["current_position"] == [0.25, 0.0, 0.0]


def test_r2r_stop_absent_pose_both_inherit():
    rows = [
        {"timestep": 0, "action": "turn left 30 degree", "current_position": [0.0, 0.0, 0.0],
         "current_heading_rad": 0.0, "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 6,
         "images": ["a.jpg"], "instruction": "x", "id": "r1"},
        {"timestep": 1, "action": "stop", "current_position": None,
         "current_heading_rad": None, "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 6,
         "images": ["b.jpg"], "instruction": "x", "id": "r2"},
    ]
    clean = r2r_rows_to_clean(rows)
    stop_row = clean[(norm_scene("ur6pFq6Qu1A"), 6)][1]
    assert stop_row["current_position"] == [0.0, 0.0, 0.0]
    assert stop_row["current_heading_rad"] == 0.0


def test_r2r_nonstop_missing_pose_drops_episode():
    rows = [
        {"timestep": 0, "action": "move forward 25 cm", "current_position": [0.0, 0.0, 0.0],
         "current_heading_rad": 0.0, "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 7,
         "images": ["a.jpg"], "instruction": "x", "id": "r1"},
        {"timestep": 1, "action": "move forward 25 cm", "current_position": None,
         "current_heading_rad": None, "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 7,
         "images": ["b.jpg"], "instruction": "x", "id": "r2"},
    ]
    assert r2r_rows_to_clean(rows) == {}


def test_r2r_first_row_stop_dropped():
    rows = [
        {"timestep": 0, "action": "stop", "current_position": None, "current_heading_rad": None,
         "source": "R2R-CE", "scene_id": "ur6pFq6Qu1A", "episode_id": 8,
         "images": ["a.jpg"], "instruction": "x", "id": "r1"},
    ]
    assert r2r_rows_to_clean(rows) == {}


def test_eval_scene_dropped():
    rows = [
        {"timestep": 0, "action": "move forward 25 cm", "current_position": [0.0, 0.0, 0.0],
         "current_heading_rad": 0.0, "source": "R2R-CE", "scene_id": "2azQ1b91cZZ", "episode_id": 9,
         "images": ["a.jpg"], "instruction": "x", "id": "r1"},
    ]
    assert r2r_rows_to_clean(rows) == {}


def test_build_excludes_yaw_and_leak_guards():
    rows = [
        _row("turn left 45 degree", t=1),
        _row("stop", t=3, images=[f"/s/frames/ep1/step{i:04d}.png" for i in range(0, 4)],
             terminal_kind="stop"),
        {"id": "y1", "source": "YawRecovery-v2", "scene_id": "ur6pFq6Qu1A", "episode_id": 1,
         "timestep": 0, "action": "turn left 45 degree", "images": ["y.jpg"], "instruction": "y"},
        {"id": "e1", "source": "R2R-CE", "scene_id": "2azQ1b91cZZ", "episode_id": 1, "timestep": 0,
         "action": "move forward 25 cm", "current_position": [0.0, 0.0, 0.0],
         "current_heading_rad": 0.0, "images": ["e.jpg"], "instruction": "e"},
    ]
    gt = {}
    for sc in ("ur6pFq6Qu1A", "2azQ1b91cZZ"):
        for ep in (1,):
            gt[(sc, ep)] = GT_EP
    points, stats = build(rows, gt)
    scenes = {p["scene_id"] for p in points}
    assert scenes == {"ur6pFq6Qu1A"}
    assert all(p["source"] not in EXCLUDED_SOURCES for p in points)
    assert len(points) == 2, [(p["id"], p["gt_action"]) for p in points]
    assert stats["terminal_kept"] == 2
    assert stats["excluded_rows"] == 1
    assert action_class("stop") == "stop"
    assert in_contract("move forward 75 cm")
    assert not in_contract("move forward 80 cm")


def test_last_step_index():
    assert last_step_index(["/a/step0007.png", "/a/step0008.png"]) == 8
    assert last_step_index([]) is None
    assert last_step_index(["/a/skybox1_sami.jpg"]) is None


def test_leak_guard_eval_scene_raises():
    try:
        from scripts.grpo.build_grpo_v2 import leak_guard
        leak_guard([{"scene_id": "2azQ1b91cZZ", "gt_action": "stop",
                     "position": [0.0, 0.0, 0.0], "heading_rad": 0.0, "images": ["a"], "id": "x"}])
        raised = False
    except AssertionError:
        raised = True
    assert raised



def test_attach_geo_after_fills_and_none():
    pts = [
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 0, "geodesic_distance": 3.0},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 1, "geodesic_distance": 2.5},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 2, "geodesic_distance": 2.0},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 2, "timestep": 0, "geodesic_distance": 5.0},
        {"scene_id": "8WUmhLawc2A", "episode_id": 3, "timestep": 0, "geodesic_distance": None},
    ]
    attach_geo_after(pts)
    assert pts[0]["geo_after"] == 2.5
    assert pts[1]["geo_after"] == 2.0
    assert pts[2]["geo_after"] is None
    assert pts[3]["geo_after"] is None
    assert pts[4]["geo_after"] is None
    assert "geo_after" in pts[4]


def test_attach_geo_after_respects_timestep_gap():
    pts = [
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 5, "geodesic_distance": 2.0},
        {"scene_id": "ur6pFq6Qu1A", "episode_id": 1, "timestep": 9, "geodesic_distance": 0.5},
    ]
    attach_geo_after(pts)
    assert pts[0]["geo_after"] is None
    assert pts[1]["geo_after"] is None


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS {n}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
