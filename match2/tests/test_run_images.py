"""採点を真似る道具 run_images.py（StubBackend）。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import run_images  # noqa: E402

SAMPLES_DIR = TOOLS.parent.parent.parent / "_src" / "mc2_samples"


def test_normalize_matches_the_organizers_rule():
    n = run_images.normalize
    assert n("7-ABC-123") == n("7 ABC 123") == n("7abc123") == "7ABC123"
    assert n("京 A·12345") == n("京 A12345") == "京A12345"
    assert n("a_b.c-d e") == "ABCDE"
    assert n("CALIFORNIA 7ABC123") != n("7ABC123") and n("PLATE: 7ABC123") != n("7ABC123")


def test_check_output_shape(tmp_path):
    f = tmp_path / "o.json"
    assert run_images.check_output_shape(f)[0] is False
    f.write_text("{")
    assert run_images.check_output_shape(f)[0] is False
    f.write_text('{"text": 3}')
    assert run_images.check_output_shape(f)[0] is False
    f.write_text('{"text": "x", "confidence": 1.5}')
    assert run_images.check_output_shape(f)[0] is False
    f.write_text('{"text": "x", "confidence": 0.5}')
    assert run_images.check_output_shape(f)[0] is True
    f.write_text('{"text": ""}')
    assert run_images.check_output_shape(f)[0] is True


def test_samples_json_matches_files():
    s = json.loads((TOOLS / "samples.json").read_text(encoding="utf-8"))["samples"]
    assert len(s) == 10 and [x["n"] for x in s] == list(range(1, 11))
    want = ["PNG", "PNG", "JPEG", "PNG", "JPEG", "PNG", "TIFF", "JPEG", "PNG", "TIFF"]  # 主催の原文が書く形
    if not SAMPLES_DIR.exists():
        pytest.skip("sample images not extracted")
    for x, f in zip(s, want):
        with Image.open(SAMPLES_DIR / x["file"]) as im:
            assert im.format == f, x


def make_small_samples(d: Path):
    d.mkdir(parents=True)
    items = []
    for n, (name, fmt) in enumerate([("a.png", "PNG"), ("b.JPG", "JPEG"), ("c.tif", "TIFF")], 1):
        Image.new("RGB", (700, 300), (200, 200, 200)).save(d / name, format=fmt)
        items.append({"n": n, "file": name, "expected": "OK 12" if n < 3 else "zz"})
    (d / "s.json").write_text(json.dumps({"samples": items}))
    return d / "s.json"


def test_run_images_with_stub_passes_gates_and_reports(short_dir):
    sj = make_small_samples(short_dir / "imgs")
    rep = short_dir / "rep.json"
    r = subprocess.run([sys.executable, str(TOOLS / "run_images.py"), "--samples-json", str(sj),
                        "--samples-dir", str(short_dir / "imgs"), "--env", "X052_BACKEND=stub",
                        "--env", "X052_STUB_ANSWER=ok-12", "--report", str(rep)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    rp = json.loads(rep.read_text())
    assert rp["startup"]["state"] == "ready" and all(v is not False for v in rp["gates"].values())
    assert rp["gates"]["vram_1_to_48gib"] is None  # この Mac には amd-smi が無い
    assert [x["match"] for x in rp["images"]] == [True, True, False]  # 一致の判定が動いている（正規化つき）
    assert all(x["valid"] and x["returncode"] == 0 for x in rp["images"])


def test_run_images_fails_the_gate_when_server_never_ready(short_dir):
    sj = make_small_samples(short_dir / "imgs")
    r = subprocess.run([sys.executable, str(TOOLS / "run_images.py"), "--samples-json", str(sj),
                        "--samples-dir", str(short_dir / "imgs"), "--env", "X052_BACKEND=stub",
                        "--env", "X052_STUB_FAIL_LOAD=1", "--env", "X052_SUPERVISE_DELAY=0.2",
                        "--env", "X052_IMAGE_BUDGET=1.5", "--env", "X052_READY_MARGIN=0.2",
                        "--startup-budget", "4"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "門 startup_time: 不合格" in r.stdout
