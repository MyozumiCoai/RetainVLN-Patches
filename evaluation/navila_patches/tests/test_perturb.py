#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "perturb_mod", Path(__file__).resolve().parents[1] / "perturb.py"
)
perturb_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perturb_mod)

PerturbInjector = perturb_mod.PerturbInjector
compute_recovery = perturb_mod.compute_recovery


class TestInjector(unittest.TestCase):
    def test_off_when_p_zero(self):
        inj = PerturbInjector(0.0, 90, "x.jsonl")
        self.assertIsNone(inj.maybe_force())

    def test_always_force_when_p_one(self):
        inj = PerturbInjector(1.0, 90, "x.jsonl")
        hits = [inj.maybe_force() for _ in range(100)]
        self.assertTrue(all(h is not None for h in hits))
        self.assertEqual(hits[0][1], 6)

    def test_deg_mapping(self):
        self.assertEqual(PerturbInjector(1.0, 15, "x").maybe_force()[1], 1)
        self.assertEqual(PerturbInjector(1.0, 45, "x").maybe_force()[1], 3)
        self.assertEqual(PerturbInjector(1.0, 180, "x").maybe_force()[1], 12)

    def test_invalid_deg_rejected(self):
        with self.assertRaises(AssertionError):
            PerturbInjector(0.5, 40, "x")
        with self.assertRaises(AssertionError):
            PerturbInjector(0.5, 0, "x")

    def test_probability_approx(self):
        inj = PerturbInjector(0.5, 90, "x", seed=42)
        n = 4000
        hits = sum(1 for _ in range(n) if inj.maybe_force() is not None)
        self.assertGreater(hits, n * 0.35)
        self.assertLess(hits, n * 0.65)

    def test_side_half_half(self):
        inj = PerturbInjector(1.0, 15, "x", seed=7)
        sides = [inj.maybe_force()[0] for _ in range(2000)]
        left = sum(1 for s in sides if s == 2)
        self.assertGreater(left, 800)
        self.assertLess(left, 1200)


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "evt.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self):
        return [json.loads(l) for l in open(self.log, encoding="utf-8") if l.strip()]

    def test_perturb_and_trajectory(self):
        inj = PerturbInjector(0.5, 90, self.log)
        inj.log_perturb(3, 5, 2, geo=8.0)
        inj.log_step(3, 6, 8.5)
        inj.log_step(3, 7, 7.2)
        inj.log_step(99, 1, 4.0)
        inj.close()
        rows = self._rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["type"], "perturb")
        self.assertEqual(rows[0]["episode_id"], 3)
        self.assertEqual(rows[0]["side"], "left")
        self.assertEqual(rows[0]["deg"], 90)
        self.assertEqual(rows[1]["type"], "step")

    def test_end_episode_stops_trajectory(self):
        inj = PerturbInjector(0.5, 90, self.log)
        inj.log_perturb(1, 2, 3, geo=5.0)
        inj.end_episode(1)
        inj.log_step(1, 3, 5.0)
        inj.close()
        self.assertEqual(len(self._rows()), 1)

    def test_stats(self):
        inj = PerturbInjector(1.0, 15, self.log)
        inj.maybe_force()
        inj.log_perturb(1, 1, 2, geo=1.0)
        self.assertEqual(inj.stats()["perturb_events"], 1)
        self.assertEqual(inj.stats()["forced_steps"], 1)


class TestRecovery(unittest.TestCase):
    def test_recovered(self):
        rows = [
            {"type": "perturb", "episode_id": 1, "step": 5, "side": "left", "deg": 90, "geo": 8.0},
            {"type": "step", "episode_id": 1, "step": 6, "geo": 8.6},
            {"type": "step", "episode_id": 1, "step": 7, "geo": 8.1},
            {"type": "step", "episode_id": 1, "step": 8, "geo": 7.4},
        ]
        r = compute_recovery(rows)
        self.assertEqual(r["n_perturb_events"], 1)
        self.assertEqual(r["n_recovered"], 1)
        self.assertEqual(r["recovery_rate"], 1.0)

    def test_not_recovered(self):
        rows = [
            {"type": "perturb", "episode_id": 1, "step": 5, "geo": 8.0},
            {"type": "step", "episode_id": 1, "step": 6, "geo": 9.0},
            {"type": "step", "episode_id": 1, "step": 7, "geo": 9.5},
            {"type": "step", "episode_id": 1, "step": 8, "geo": 10.0},
        ]
        r = compute_recovery(rows)
        self.assertEqual(r["n_recovered"], 0)
        self.assertEqual(r["recovery_rate"], 0.0)

    def test_window_boundary(self):
        rows = [{"type": "perturb", "episode_id": 1, "step": 0, "geo": 8.0}]
        for t in range(1, 12):
            rows.append({"type": "step", "episode_id": 1, "step": t, "geo": 8.0 + t * 0.05})
        rows.append({"type": "step", "episode_id": 1, "step": 11, "geo": 7.0})
        r = compute_recovery(rows)
        self.assertEqual(r["n_recovered"], 0)

    def test_multi_episode_mixed(self):
        rows = [
            {"type": "perturb", "episode_id": 1, "step": 0, "geo": 8.0},
            {"type": "step", "episode_id": 1, "step": 1, "geo": 7.0},
            {"type": "perturb", "episode_id": 2, "step": 0, "geo": 5.0},
            {"type": "step", "episode_id": 2, "step": 1, "geo": 6.0},
        ]
        r = compute_recovery(rows)
        self.assertEqual(r["n_perturb_events"], 2)
        self.assertEqual(r["n_recovered"], 1)
        self.assertEqual(r["recovery_rate"], 0.5)
        self.assertEqual(r["n_episodes_perturbed"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
