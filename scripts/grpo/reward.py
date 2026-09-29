from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

STOP = "stop"
FWD = "forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"

FWD_GRID = (25.0, 50.0, 75.0)
TURN_GRID = (15.0, 30.0, 45.0)

CONTRACT_RE = re.compile(
    r"^(?:stop|move forward (?:25|50|75) cm|turn (?:left|right) (?:15|30|45) degree)$"
)

_PREFIX_RE = re.compile(r"the next action is\s*", re.IGNORECASE)
_TURN_RE = re.compile(
    r"turn\s+(left|right)\s+(?:by\s+)?(-?[0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|deg|°)?",
    re.IGNORECASE,
)
_FWD_RE = re.compile(
    r"move\s+forward\s+(-?[0-9]+(?:\.[0-9]+)?)\s*(cm|centimeters?|centimetres?|meters?|metres?|m)?",
    re.IGNORECASE,
)
_STOP_RE = re.compile(r"\bstop\b", re.IGNORECASE)

Action = Tuple[str, Optional[float]]


def parse_action(text: object) -> Optional[Action]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    s = _PREFIX_RE.sub("", s).strip()
    s = s.rstrip(".").strip()
    if not s:
        return None

    m_turn = _TURN_RE.search(s)
    if m_turn:
        side = m_turn.group(1)
        deg = float(m_turn.group(2))
        return (TURN_LEFT if side == "left" else TURN_RIGHT, deg)

    m_fwd = _FWD_RE.search(s)
    if m_fwd:
        val = float(m_fwd.group(1))
        unit = (m_fwd.group(2) or "cm").lower()
        if unit.startswith("m"):
            val *= 100.0
        return (FWD, val)

    if _STOP_RE.search(s):
        return (STOP, None)

    return None


def in_contract(action: Optional[Action], tol: float = 1e-6) -> bool:
    if action is None:
        return False
    t, p = action
    if t == STOP:
        return True
    if p is None:
        return False
    if t == FWD:
        return any(abs(p - g) <= tol for g in FWD_GRID)
    if t in (TURN_LEFT, TURN_RIGHT):
        return any(abs(p - g) <= tol for g in TURN_GRID)
    return False


def reward(
    pred_text: object,
    gt_action: object,
    *,
    w_type: float = 1.0,
    w_param: float = 0.5,
    w_stop: float = 1.0,
    w_fmt: float = 0.5,
    fwd_scale: float = 25.0,
    turn_scale: float = 15.0,
    geo_t: Optional[float] = None,
    geo_after: Optional[float] = None,
    w_prog: float = 1.0,
    w_stop_ok: float = 1.0,
    w_stop_soft: float = 0.5,
    w_stop_early: float = 1.0,
    radius_ok: float = 1.5,
    radius_far: float = 3.0,
) -> float:
    gt = parse_action(gt_action)
    if gt is None:
        raise ValueError(f"GT 动作不可解析: {gt_action!r}")
    pred = parse_action(pred_text)
    if pred is None:
        return -w_fmt

    r = 0.0
    if not in_contract(pred):
        r -= w_fmt

    p_t, p_p = pred
    g_t, g_p = gt
    if p_t == g_t:
        r += w_type
        if g_t == STOP:
            r += w_stop
        else:
            scale = fwd_scale if g_t == FWD else turn_scale
            r += w_param * math.exp(-abs((p_p or 0.0) - (g_p or 0.0)) / scale)

    if geo_t is not None:
        if p_t == STOP:
            if geo_t < radius_ok:
                r += w_stop_ok * (1.0 - geo_t / radius_ok)
            elif geo_t < radius_far:
                r -= w_stop_soft * (geo_t - radius_ok) / (radius_far - radius_ok)
            else:
                r -= w_stop_early
        elif p_t == FWD and g_t == FWD and geo_after is not None:
            d_gt, d_pred = (g_p or 0.0), (p_p or 0.0)
            drop = geo_t - geo_after
            if d_gt > 0.0 and drop > 0.0:
                prog = min(drop / (d_gt / 100.0), 1.0) * min(d_pred / d_gt, 1.0)
                r += w_prog * max(prog, 0.0)
    return r


def normalize_group(rewards: Sequence[float], eps: float = 1e-4) -> Tuple[List[float], bool]:
    vals = [float(x) for x in rewards]
    n = len(vals)
    if n == 0:
        return [], True
    if max(vals) - min(vals) < eps:
        return [0.0] * n, True
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)
    adv = [(x - mean) / (std + 1e-8) for x in vals]
    return adv, False
