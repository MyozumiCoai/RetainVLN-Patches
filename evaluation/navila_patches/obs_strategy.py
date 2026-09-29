from __future__ import annotations

import os
import copy
import numpy as np
from PIL import Image



VALID_MODES = ("linspace", "recency", "random_uniform", "three_criteria")


def _cfg():
    return {
        "mode":   os.environ.get("NAVI_OBS_MODE", "linspace"),
        "gamma":  float(os.environ.get("NAVI_OBS_GAMMA", "0.0")),
        "budget": int(os.environ.get("NAVI_OBS_BUDGET", "8")),
        "seed":   int(os.environ.get("NAVI_OBS_SEED", "0")),
    }



def sample_and_pad_images(images, num_frames: int = 8, width: int = 512, height: int = 512,
                          *, mode: str | None = None, gamma: float | None = None,
                          budget: int | None = None, seed: int | None = None) -> list:
    cfg = _cfg()
    mode   = mode   if mode   is not None else cfg["mode"]
    gamma  = gamma  if gamma  is not None else cfg["gamma"]
    budget = budget if budget is not None else cfg["budget"]
    seed   = seed   if seed   is not None else cfg["seed"]

    if mode not in VALID_MODES:
        raise ValueError(f"NAVI_OBS_MODE={mode!r} 不合法，可选: {VALID_MODES}")
    if budget < 1:
        raise ValueError(f"NAVI_OBS_BUDGET={budget} 必须 ≥1")

    rng = np.random.default_rng(seed)
    frames = copy.deepcopy(images)

    if len(frames) < budget:
        while len(frames) < budget:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

    k = budget - 1
    latest = frames[-1]

    if mode == "linspace":
        idx = np.linspace(0, len(frames) - 1, k, endpoint=False, dtype=int)
        return [frames[i] for i in idx] + [latest]

    pool = frames[:-1]

    if len(pool) <= budget - 1:
        return list(pool) + [latest]

    if mode == "recency":
        idx = _recency_indices(len(pool), k, gamma, rng)
    elif mode == "random_uniform":
        idx = _random_uniform_indices(len(pool), k, rng)
    elif mode == "three_criteria":
        idx = _three_criteria_indices(pool, k, rng)
    else:
        raise ValueError(f"NAVI_OBS_MODE={mode!r} 不合法，可选: {VALID_MODES}")

    sampled = [pool[i] for i in idx] + [latest]
    return sampled



def _recency_indices(n: int, k: int, gamma: float, rng: np.random.Generator) -> np.ndarray:
    if gamma <= 0 or n <= k:
        return _random_uniform_indices(n, k, rng)
    idx = np.arange(n)
    w = np.exp(gamma * idx / max(n - 1, 1))
    w = w / w.sum()
    chosen = rng.choice(n, size=k, replace=False, p=w)
    chosen.sort()
    return chosen


def _random_uniform_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    chosen = rng.choice(n, size=k, replace=False)
    chosen.sort()
    return chosen


def _three_criteria_indices(pool: list, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(pool)
    if n <= k:
        return np.linspace(0, n - 1, n, endpoint=False, dtype=int) if n > k else np.arange(n)

    thumbs = []
    for im in pool:
        g = im.convert("L").resize((64, 64))
        thumbs.append(np.asarray(g, dtype=np.int16))
    thumbs = np.stack(thumbs, axis=0)

    scores = np.zeros(n, dtype=np.float32)
    if n > 1:
        diffs = np.abs(np.diff(thumbs, axis=0)).mean(axis=(1, 2))
        scores[1:] = diffs
    scores[0] = np.abs(thumbs[0] - thumbs[1]).mean() if n > 1 else 0.0

    topk = np.argsort(-scores)[:k]
    topk.sort()
    return topk



def describe() -> str:
    cfg = _cfg()
    return (f"obs_strategy(mode={cfg['mode']}, gamma={cfg['gamma']}, "
            f"budget={cfg['budget']}, seed={cfg['seed']})")
