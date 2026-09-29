import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from scripts.m4_memory.region import RegionGrid
from scripts.m4_memory.events import detect_turn, loop_detected
from scripts.m4_memory.manager import EpisodeMemory, EvidenceNotebook, MemoryManager
from scripts.m4_memory.text import build_memory_context, truncate


def test_region_new_and_loop():
    g = RegionGrid(cell_size=2.0)
    assert g.mark(0.0, 0.0, 0) is True
    assert g.mark(0.5, 0.5, 1) is False
    assert g.mark(3.0, 0.0, 2) is True
    assert g.cell(3.0, 0.0) == (1, 0)


def test_turn_detection():
    assert abs(detect_turn(0.0, math.pi / 2) - 90.0) < 1e-6
    assert detect_turn(0.0, math.radians(10.0)) == 0.0


def test_loop_detected():
    g = RegionGrid(cell_size=2.0)
    g.mark(0.0, 0.0, 0)
    g.mark(5.0, 0.0, 10)
    assert loop_detected(g, 0.0, 0.0, 50, min_gap=30)
    g.mark(0.0, 0.0, 50)
    assert not loop_detected(g, 8.0, 0.0, 55, min_gap=30)
    assert not loop_detected(g, 0.0, 0.0, 51, min_gap=30)


def test_episode_salient_cap():
    m = EpisodeMemory(max_salient=3)
    for i in range(5):
        m.add_salient(f"事件{i}")
    assert m.salient == ["事件2", "事件3", "事件4"]


def test_manager_context():
    mgr = MemoryManager(budget_tokens=400)
    for step in range(12):
        x = step * 0.25
        mgr.update([x, 0.0, x], 0.0, step)
    ctx = mgr.context_text()
    assert "Progress" in ctx or "Events" in ctx
    assert len(ctx) <= int(400 * 3.5) + 10
    mgr.update([10.0, 0.0, 10.0], 0.0, 12, done_info={"success": 1.0})
    assert mgr.episode.outcome == "成功到达"
    assert mgr.notebook.searched_regions


def test_context_budget():
    ctx = build_memory_context(
        {"progress": "x" * 5000, "salient": ["a", "b"], "outcome": "进行中"},
        {"searched_regions": ["区域(0,0)"] * 10, "candidate_locations": [], "rejected_hypotheses": []},
        budget_tokens=200,
    )
    assert len(ctx) <= int(200 * 3.5) + 10
    assert "..." in ctx


def test_no_oracle_leak_in_context():
    mgr = MemoryManager(budget_tokens=400)
    for step in range(30):
        x = step * 0.5
        mgr.update([x, 0.0, x], 0.0, step)
    mgr.update([0.0, 0.0, 0.0], 0.0, 31)
    ctx = mgr.context_text()
    assert "测地" not in ctx
    assert "接近目标" not in ctx


def main():
    tests = [
        test_region_new_and_loop,
        test_turn_detection,
        test_loop_detected,
        test_episode_salient_cap,
        test_manager_context,
        test_context_budget,
        test_no_oracle_leak_in_context,
    ]
    for t in tests:
        t()
        print(f"[{t.__name__}] PASS")
    print(f"result: {len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
