#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path


def _normalize(rec: dict) -> dict:
    return {
        "distance_to_goal": rec.get("distance_to_goal", rec.get("ne")),
        "success": rec["success"],
        "spl": rec["spl"],
        "oracle_success": rec.get("oracle_success", rec.get("os")),
        "path_length": rec.get("path_length", rec.get("tl", rec.get("traj_len", 0))),
        "steps_taken": rec.get("steps_taken", rec.get("steps", 0)),
    }


def load(path: str):
    raw = open(path, encoding="utf-8-sig").read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if isinstance(data, dict):
        records = list(data.values())
    else:
        records = data
    return [_normalize(r) for r in records]


def metrics(records: list) -> dict:
    d = {i: v for i, v in enumerate(records)}
    n = len(d)
    sr = sum(v["success"] for v in d.values()) / n
    spl = sum(v["spl"] for v in d.values()) / n
    osr = sum(v["oracle_success"] for v in d.values()) / n
    ne = sum(v["distance_to_goal"] for v in d.values()) / n
    tl = sum(v.get("path_length", 0) for v in d.values()) / n

    fail = [v for v in d.values() if v["success"] < 0.5]
    os0 = sum(1 for v in fail if v["oracle_success"] < 0.5)
    os1 = len(fail) - os0

    ne10 = sum(1 for v in d.values() if v["distance_to_goal"] > 10)

    return dict(
        n=n, sr=sr, spl=spl, osr=osr, ne=ne, tl=tl,
        fail=len(fail),
        os0=os0, os1=os1,
        os0_pct=os0 / len(fail) if fail else 0.0,
        os1_pct=os1 / len(fail) if fail else 0.0,
        ne10=ne10, ne10_pct=ne10 / n,
        osr_to_sr=(sr / osr) if osr > 0 else 0.0,
    )


def fmt(m: dict) -> str:
    L = []
    L.append(f"episodes        : {m['n']}")
    L.append(f"SR  (success)   : {m['sr']*100:.1f}%")
    L.append(f"SPL             : {m['spl']*100:.1f}%")
    L.append(f"OSR (oracle)    : {m['osr']*100:.1f}%")
    L.append(f"NE  (goal dist) : {m['ne']:.2f} m")
    L.append(f"TL  (path len)  : {m['tl']:.2f} m")
    L.append(f"failures        : {m['fail']} ({m['fail']/m['n']*100:.1f}%)")
    L.append(f"  OS=0 迷路      : {m['os0']} ({m['os0_pct']*100:.1f}% of fail)")
    L.append(f"  OS=1 不停      : {m['os1']} ({m['os1_pct']*100:.1f}% of fail)")
    L.append(f"  NE>10m         : {m['ne10']} ({m['ne10_pct']*100:.1f}%)")
    if m["osr"] > 0:
        L.append(f"  OSR->SR 转化   : {m['osr_to_sr']*100:.1f}%")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="M5 VLN 失败模式评测工具")
    ap.add_argument("result", help="result json (per-episode metrics)")
    ap.add_argument("--compare", help="可选 baseline json，输出增量对比")
    ap.add_argument("--out", help="可选 markdown 报告输出路径")
    args = ap.parse_args()

    m = metrics(load(args.result))
    report = f"# 评测指标\n\n`{args.result}`\n\n```\n{fmt(m)}\n```\n"

    if args.compare:
        b = metrics(load(args.compare))
        dsr = (m["sr"] - b["sr"]) * 100
        dne = m["ne"] - b["ne"]
        dosr = (m["osr"] - b["osr"]) * 100
        report += (
            f"\n## 对比基线 `{args.compare}`\n\n"
            f"ΔSR={dsr:+.1f}pp, ΔSPL={(m['spl']-b['spl'])*100:+.1f}pp, "
            f"ΔOSR={dosr:+.1f}pp, ΔNE={dne:+.2f}m\n\n"
            f"```\n{fmt(b)}\n```\n"
        )

    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
