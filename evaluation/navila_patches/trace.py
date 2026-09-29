from __future__ import annotations

import json
import math


def softmax_entropy_first(logits) -> dict:
    try:
        import torch

        z = torch.as_tensor(logits, dtype=torch.float32).flatten()
        m = z.max()
        e = (z - m).exp()
        s = e.sum()
        p = e / s
        ent = -(p * p.log()).sum().item()
        return {"ent_first": round(float(ent), 4), "top1_p": round(float(p.max().item()), 4)}
    except Exception:
        return None


class TraceRecorder:

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.n = 0

    def write(self, **row) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass
