import os
import random
import sys
import tempfile

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from scripts.grpo.trainer import batch_action_dist, resolve_nav_path, select_batch


def test_resolve_windows_scans_anchored():
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "vyrNrziPKCB", "vyrNrziPKCB", "matterport_skybox_images")
        os.makedirs(d)
        fp = os.path.join(d, "abc_skybox1_sami.jpg")
        open(fp, "w").close()
        win = r"F:\MP3D\v1\scans\vyrNrziPKCB\vyrNrziPKCB\matterport_skybox_images\abc_skybox1_sami.jpg"
        assert resolve_nav_path(win, root) == fp


def test_resolve_first_scans_anchor():
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "sceneA", "imgs")
        os.makedirs(d)
        fp = os.path.join(d, "f.jpg")
        open(fp, "w").close()
        p = "F:/MP3D/v1/scans/extra/scans/sceneA/imgs/f.jpg"
        assert resolve_nav_path(p, root) is None
        p2 = "F:/MP3D/v1/scans/sceneA/imgs/f.jpg"
        assert resolve_nav_path(p2, root) == fp


def test_resolve_posix_absolute_existing():
    with tempfile.TemporaryDirectory() as root:
        fp = os.path.join(root, "abs.jpg")
        open(fp, "w").close()
        assert resolve_nav_path(fp, "/some/other/root") == fp


def test_resolve_posix_absolute_missing():
    assert resolve_nav_path("/nonexistent/does/not/exist.jpg", "/some_root") is None


def test_resolve_no_scans_drops_first_component():
    with tempfile.TemporaryDirectory() as root:
        d = os.path.join(root, "sceneB")
        os.makedirs(d)
        fp = os.path.join(d, "g.jpg")
        open(fp, "w").close()
        p = "F:/sceneB/g.jpg"
        assert resolve_nav_path(p, root) == fp


def test_resolve_no_root_relative_none():
    assert resolve_nav_path("F:/MP3D/v1/scans/s/f.jpg", None) is None


def test_resolve_missing_file_none():
    with tempfile.TemporaryDirectory() as root:
        assert resolve_nav_path("F:/MP3D/v1/scans/nope/nope.jpg", root) is None


def test_select_batch_no_dup():
    s = [{"id": i} for i in range(10)]
    b = select_batch(s, 4, random.Random(0))
    assert len(b) == 4 and len({x["id"] for x in b}) == 4


def test_select_batch_ge_len_returns_all():
    s = [{"id": i} for i in range(3)]
    assert len(select_batch(s, 99, random.Random(0))) == 3


def test_batch_action_dist():
    d = batch_action_dist([
        "The next action is move forward 75 cm.",
        "stop",
        "turn left 30 degree",
        "turn right 45 degree",
        "garbage",
    ])
    assert d.get("forward") == 1
    assert d.get("stop") == 1
    assert d.get("turn_left") == 1
    assert d.get("turn_right") == 1
    assert d.get("unparsed") == 1


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        fn()
        passed += 1
        print(f"  PASS {name}")
    print(f"\n[test_trainer_pure] {passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
