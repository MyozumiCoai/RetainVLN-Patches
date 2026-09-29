from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

CONTRACT_RE = re.compile(
    r"^(?:stop|move forward (?:25|50|75) cm|turn (?:left|right) (?:15|30|45) degree)$"
)
EVAL_SCENES = {"2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "QUCTc6BB5sX", "TbHJrupSAjP"}
EXCLUDED_SOURCES = {"YawRecovery-v2"}

EpKey = Tuple[str, object]
STEP_RE = re.compile(r"step(\d+)")


def norm_scene(s: object) -> str:
    s = str(s).strip()
    if "/" in s:
        s = s.rstrip("/").split("/")[-1]
    if s.endswith(".glb"):
        s = s[: -len(".glb")]
    return s


def in_contract(a: object) -> bool:
    return bool(CONTRACT_RE.match(str(a or "").strip()))


def has_pose(r: dict) -> bool:
    return r.get("current_position") is not None and r.get("current_heading_rad") is not None


def action_class(a: object) -> str:
    a = str(a or "")
    if a == "stop":
        return "stop"
    if a.startswith("move"):
        return "forward"
    if a.startswith("turn"):
        return "turn"
    return "?"


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_r2r_gt(path: str) -> Dict[EpKey, dict]:
    if path.endswith(".gz"):
        with open(path, "rb") as f:
            d = json.loads(gzip.decompress(f.read()).decode("utf-8"))
    else:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    eps = d["episodes"] if isinstance(d, dict) and "episodes" in d else d
    out = {}
    for e in eps:
        out[(norm_scene(e.get("scene_id") or e.get("scan", "?")), e.get("episode_id"))] = e
    return out


def last_step_index(images: Optional[List[str]]) -> Optional[int]:
    if not images:
        return None
    m = None
    for p in reversed(images):
        m = STEP_RE.search(str(p))
        if m:
            break
    return int(m.group(1)) if m else None


def densify_gt(gt_ep: dict) -> List[dict]:
    try:
        from scripts.t3_terminal.render_terminal import densify_path, extend_path_with_goal
    except ImportError:
        from t3_terminal.render_terminal import densify_path, extend_path_with_goal
    path = gt_ep.get("reference_path") or []
    goals = gt_ep.get("goals") or []
    if not path or not goals:
        return []
    goal = goals[0]["position"]
    return densify_path(extend_path_with_goal(path, goal))


def terminal_row_to_point(r: dict, gt_ep: dict, subs: List[dict]) -> Optional[dict]:
    t = int(r["timestep"])
    if t < 0 or t >= len(subs):
        return None
    if not in_contract(r.get("action")):
        return None
    pose = subs[t]
    goal = (gt_ep.get("goals") or [{}])[0].get("position")
    fi = last_step_index(r.get("images"))
    if fi is not None and fi != t:
        return None
    return {
        "id": r.get("id"),
        "source": r.get("source"),
        "scene_id": norm_scene(r["scene_id"]),
        "episode_id": r["episode_id"],
        "timestep": t,
        "ep_len": len(subs),
        "instruction": r.get("instruction", ""),
        "images": r.get("images", []),
        "gt_action": str(r["action"]).strip(),
        "action_class": action_class(r.get("action")),
        "position": pose["position"],
        "heading_rad": pose["heading_rad"],
        "prefix_poses": [
            [*subs[k]["position"], subs[k]["heading_rad"]] for k in range(t + 1)
        ],
        "direct_answer": r.get("direct_answer") or f"The next action is {str(r['action']).strip()}.",
        "geodesic_distance": r.get("geodesic_to_goal"),
        "next_position": subs[t + 1]["position"] if t + 1 < len(subs) else None,
        "goal_position": goal,
    }


def r2r_rows_to_clean(rows: List[dict]) -> Dict[EpKey, List[dict]]:
    by_ep: Dict[EpKey, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        t = r.get("timestep")
        if t is None:
            continue
        by_ep[(norm_scene(r.get("scene_id")), r.get("episode_id"))][int(t)] = r
    out: Dict[EpKey, List[dict]] = {}
    for k, ts_map in by_ep.items():
        if norm_scene(k[0]) in EVAL_SCENES:
            continue
        ts = sorted(ts_map)
        if not ts or ts[0] != 0 or ts != list(range(len(ts))):
            continue
        ep_rows = [ts_map[t] for t in ts]
        ok = True
        for i, r in enumerate(ep_rows):
            a = str(r.get("action") or "").strip()
            if not in_contract(a):
                ok = False
                break
            if not has_pose(r):
                if a == "stop" and i > 0 and has_pose(ep_rows[i - 1]):
                    prev = ep_rows[i - 1]
                    r = dict(r)
                    if r.get("current_position") is None:
                        r["current_position"] = prev["current_position"]
                    if r.get("current_heading_rad") is None:
                        r["current_heading_rad"] = prev["current_heading_rad"]
                    ep_rows[i] = r
                else:
                    ok = False
                    break
        if ok:
            out[k] = ep_rows
    return out


def r2r_row_to_point(r: dict, scene: str, ep, *, ep_len: int) -> dict:
    meta = r.get("metadata") or {}
    return {
        "id": r.get("id"),
        "source": r.get("source"),
        "scene_id": scene,
        "episode_id": ep,
        "timestep": int(r["timestep"]),
        "ep_len": ep_len,
        "instruction": r.get("instruction", ""),
        "images": r.get("images", []),
        "gt_action": str(r["action"]).strip(),
        "action_class": action_class(r.get("action")),
        "position": r.get("current_position"),
        "heading_rad": r.get("current_heading_rad"),
        "direct_answer": r.get("direct_answer") or f"The next action is {str(r['action']).strip()}.",
        "geodesic_distance": r.get("geodesic_distance"),
        "next_position": r.get("next_position"),
        "goal_position": meta.get("goal_position"),
    }


def attach_geo_after(points: List[dict], extra_rows: Optional[List[dict]] = None) -> None:
    idx: Dict[Tuple[str, object], Dict[int, float]] = defaultdict(dict)
    for p in points:
        g = p.get("geodesic_distance")
        if g is not None:
            idx[(str(p["scene_id"]), p["episode_id"])][int(p["timestep"])] = g
    for r in extra_rows or []:
        g = r.get("geodesic_to_goal")
        if g is not None:
            idx[(norm_scene(r.get("scene_id")), r.get("episode_id"))][int(r["timestep"])] = g
    n_filled = 0
    for p in points:
        nxt = idx[(str(p["scene_id"]), p["episode_id"])].get(int(p["timestep"]) + 1)
        p["geo_after"] = nxt
        n_filled += nxt is not None
    print(f"[geo_after] 覆盖率 {n_filled}/{len(points)} ({n_filled / max(1, len(points)):.1%})")


def leak_guard(points: List[dict]) -> None:
    scenes = {norm_scene(p["scene_id"]) for p in points}
    bad = scenes & EVAL_SCENES
    assert not bad, f"出现评测场景: {sorted(bad)}"
    for p in points:
        assert in_contract(p["gt_action"]), f"动作超契约: {p['gt_action']!r} {p.get('id')}"
        assert p["position"] is not None and p["heading_rad"] is not None, f"缺 pose: {p.get('id')}"
        assert p["images"], f"缺帧: {p.get('id')}"


def build(src_rows: List[dict], gt: Dict[EpKey, dict], *, terminal_raw: Optional[List[dict]] = None) -> Tuple[List[dict], dict]:
    r2r_rows = [r for r in src_rows if r.get("source") in ("R2R-CE", "RxR-CE")]
    term_rows = [r for r in src_rows if r.get("source") == "Terminal-v1"]
    excluded = sum(1 for r in src_rows if r.get("source") in EXCLUDED_SOURCES)

    clean_eps = r2r_rows_to_clean(r2r_rows)
    points: List[dict] = []
    n_clean_ep = 0
    for (sc, ep), rows in clean_eps.items():
        prefix: List[list] = []
        for r in rows:
            prefix.append([*r["current_position"], r["current_heading_rad"]])
            pt = r2r_row_to_point(r, sc, ep, ep_len=len(rows))
            pt["prefix_poses"] = list(prefix)
            points.append(pt)
        n_clean_ep += 1

    n_term = 0
    n_term_no_gt = 0
    n_term_misaligned = 0
    term_by_src = Counter()
    for r in term_rows:
        sc = norm_scene(r["scene_id"])
        k = (sc, r["episode_id"])
        if k not in gt:
            n_term_no_gt += 1
            continue
        subs = densify_gt(gt[k])
        if not subs:
            n_term_no_gt += 1
            continue
        p = terminal_row_to_point(r, gt[k], subs)
        if p is None:
            n_term_misaligned += 1
            continue
        points.append(p)
        term_by_src[str(r.get("action")).strip().split()[0]] += 1
        n_term += 1

    points.sort(key=lambda p: (str(p["scene_id"]), str(p["episode_id"]), p["timestep"]))
    attach_geo_after(points, extra_rows=terminal_raw)
    leak_guard(points)
    cls_dist = Counter(p["action_class"] for p in points)
    src_dist = Counter(p["source"] for p in points)
    stats = {
        "src_rows": len(src_rows),
        "excluded_rows": excluded,
        "r2r_rows": len(r2r_rows),
        "clean_episodes": n_clean_ep,
        "terminal_rows": len(term_rows),
        "terminal_kept": n_term,
        "terminal_no_gt": n_term_no_gt,
        "terminal_misaligned": n_term_misaligned,
        "terminal_kept_dist": dict(term_by_src),
        "points": len(points),
        "class_dist": dict(cls_dist),
        "source_dist": dict(src_dist),
        "scene_dist": dict(Counter(norm_scene(p["scene_id"]) for p in points)),
        "stop_frac": round(cls_dist.get("stop", 0) / max(1, len(points)), 4),
        "all_have_pose": True,
        "all_in_contract": True,
        "all_have_frames": True,
    }
    return points, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="t3_sft_mix.jsonl")
    ap.add_argument("--gt", required=True, help="R2R train episodes json(.gz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--terminal-raw", default=None, help="Terminal 渲染全量 jsonl(补 geo_after 空洞)")
    args = ap.parse_args()

    src_rows = read_jsonl(args.src)
    gt = read_r2r_gt(args.gt)
    extra = read_jsonl(args.terminal_raw) if args.terminal_raw else None
    points, stats = build(src_rows, gt, terminal_raw=extra)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "grpo_train_v2.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "grpo_v2_build.stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[src] {stats['src_rows']}（排除 yaw {stats['excluded_rows']}）")
    print(f"[r2r/rxr clean] episodes={stats['clean_episodes']}")
    print(f"[terminal] kept={stats['terminal_kept']}/{stats['terminal_rows']} "
          f"(no_gt={stats['terminal_no_gt']} misalign={stats['terminal_misaligned']})")
    print(f"[total] points={stats['points']} class={stats['class_dist']} "
          f"stop_frac={stats['stop_frac']} src={stats['source_dist']}")
    print(f"[场景] {len(stats['scene_dist'])} 个: {stats['scene_dist']}")
    print(f"[防泄漏] 评测场景=0 契约=全过 pose=全有 帧=全有 ✓")
    print(f"[输出] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
