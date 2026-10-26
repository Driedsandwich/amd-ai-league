"""層の作成・送り・照らしの試験。crane と手元の登録所（127.0.0.1）と alpine を使う。
crane の場所は環境変数 X052_CRANE。無ければ crane を使う試験は skip。"""
import hashlib
import os
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(TOOLS))

import make_layers  # noqa: E402
import push_image  # noqa: E402
import verify_image  # noqa: E402

CRANE = os.environ.get("X052_CRANE")
needs_crane = pytest.mark.skipif(not CRANE or not os.path.exists(CRANE), reason="X052_CRANE not set")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def trees(tmp_path):
    site = tmp_path / "site" / "pkg"
    site.mkdir(parents=True)
    (site / "m.py").write_text("x=1\n")
    models = tmp_path / "models" / "tiny"
    models.mkdir(parents=True)
    (models / "w1.bin").write_bytes(os.urandom(200000))
    (models / "w2.bin").write_bytes(os.urandom(100000))
    (models / "w3.bin").write_bytes(os.urandom(50000))
    return tmp_path / "site", tmp_path / "models"


def test_make_layers_is_deterministic(trees, tmp_path):
    site, models = trees
    r1 = make_layers.make_layers(tmp_path / "o1", APP, site, models, weights_max_bytes=250000)
    os.utime(site / "pkg" / "m.py", (12345, 12345))  # 時刻が違っても同じ tar
    r2 = make_layers.make_layers(tmp_path / "o2", APP, site, models, weights_max_bytes=250000)
    a = [Path(l["path"]).name for l in r1["layers"]]
    assert a == [Path(l["path"]).name for l in r2["layers"]]
    for l1, l2 in zip(r1["layers"], r2["layers"]):
        assert sha(l1["path"]) == sha(l2["path"]), l1["name"]
    with tarfile.open(r1["layers"][0]["path"]) as tf:
        members = tf.getmembers()
    names = [m.name for m in members]
    assert {"app", "app/corpus", "app/output", "app/index", "app/app.py", "app/x052/answerer.py"} <= set(names)
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)
    assert all(m.uid == 0 and m.gid == 0 and m.mtime == make_layers.FIXED_MTIME for m in members)
    assert names == sorted(names)
    with tarfile.open(r1["layers"][1]["path"]) as tf:
        assert "opt/x052/site/pkg/m.py" in tf.getnames()


def test_weights_split_at_cap(trees, tmp_path):
    site, models = trees
    r = make_layers.make_layers(tmp_path / "o", None, None, models, weights_max_bytes=250000)
    w = [l for l in r["layers"] if l["name"].startswith("weights")]
    assert len(w) == 2 and all(l["uncompressed_bytes"] < 250000 for l in w)
    assert sum(l["uncompressed_bytes"] for l in w) == 350000
    with tarfile.open(w[0]["path"]) as tf:
        assert "models/tiny/w1.bin" in tf.getnames()
    # 上限が大きければ 1 層
    r = make_layers.make_layers(tmp_path / "o3", None, None, models)
    assert len([l for l in r["layers"] if l["name"].startswith("weights")]) == 1
    # 1 ファイルが上限以上なら例外
    with pytest.raises(ValueError):
        make_layers.make_layers(tmp_path / "o4", None, None, models, weights_max_bytes=100000)


def test_size_budget_math(trees, tmp_path):
    site, models = trees
    r = make_layers.make_layers(tmp_path / "o", APP, None, None, base_gib=59.999999, limit_gib=60)
    assert r["fits"] in (True, False) and abs(r["total_gib"] - (59.999999 + r["own_bytes"] / 1024 ** 3)) < 1e-6
    r = make_layers.make_layers(tmp_path / "o5", APP, None, None, base_gib=60.0)
    assert r["fits"] is False and r["headroom_gib"] < 0


def test_push_refuses_non_local_without_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("X052_IMAGE_REF", "registry.example.invalid/ns/img:v1")
    layer = tmp_path / "l.tar"
    layer.write_bytes(b"")
    r = subprocess.run([sys.executable, str(TOOLS / "push_image.py"), "-f", str(layer), "--model-name", "m",
                        "--crane", "/nonexistent/crane"], capture_output=True, text=True)
    assert r.returncode == 3 and "承認" in r.stderr
    assert "registry.example.invalid" not in (r.stdout + r.stderr)  # 送り先を書き出さない
    r = subprocess.run([sys.executable, str(TOOLS / "push_image.py"), "-f", str(layer), "--model-name", "m",
                        "--local", "--crane", "/nonexistent/crane"], capture_output=True, text=True,
                       env=dict(os.environ, X052_IMAGE_REF="registry.example.invalid/ns/img:v1"))
    assert r.returncode == 2  # --local は手元の登録所だけ


@pytest.fixture
def registry():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    p = subprocess.Popen([CRANE, "registry", "serve", "--address", f"127.0.0.1:{port}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        yield f"127.0.0.1:{port}"
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


@needs_crane
def test_push_and_verify_end_to_end_and_wrong_base_fails(trees, tmp_path, registry, monkeypatch):
    site, models = trees
    out = tmp_path / "layers"
    r = make_layers.make_layers(out, APP, site, models, weights_max_bytes=250000, base_gib=0.01)
    tars = [l["path"] for l in r["layers"]]
    ref = f"{registry}/x052/test:v1"
    monkeypatch.setenv("X052_IMAGE_REF", ref)
    args = [x for t in tars for x in ("-f", t)]
    p = subprocess.run([sys.executable, str(TOOLS / "push_image.py"), "--local", "--model-name", "tiny",
                        "--crane", CRANE, *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert ref not in p.stdout + p.stderr
    good = verify_image.verify(ref, "alpine:3.20", CRANE, insecure=True, layer_tars=tars, base_gib=0.01,
                               model_name="tiny")
    assert all(ok for _, ok, _ in good), [x for x in good if not x[1]]
    assert len(good) >= 12
    # 違う土台で照らすと落ちる（照らす道具が落とせる）
    bad = verify_image.verify(ref, "alpine:3.19", CRANE, insecure=True, layer_tars=tars, base_gib=0.01,
                              model_name="tiny")
    failed = {n for n, ok, _ in bad if not ok}
    assert "目録の下の層が土台と同じ（圧縮の識別子）" in failed
    assert "config の diff_ids の下が土台と同じ" in failed
    # 合計が上限を超えるなら落ちる
    big = verify_image.verify(ref, "alpine:3.20", CRANE, insecure=True, layer_tars=tars, base_gib=60.0,
                              model_name="tiny")
    assert "合計が 60GiB に収まる" in {n for n, ok, _ in big if not ok}
    # 環境変数の違い（別のモデル名）でも落ちる
    wrong = verify_image.verify(ref, "alpine:3.20", CRANE, insecure=True, layer_tars=tars, base_gib=0.01,
                                model_name="other")
    assert "自分の環境変数が足されている" in {n for n, ok, _ in wrong if not ok}


# ---- 起動の命令を渡す旗（--cmd。試合 2 の箱が使う）。既定は今までと同じ sleep infinity ----
def test_mutate_flags_default_is_unchanged_sleep_infinity():
    f = push_image.mutate_flags("m")
    assert f[:6] == ["--cmd", "sleep", "--cmd", "infinity", "-w", "/app"]
    assert f == push_image.mutate_flags("m", None) == push_image.mutate_flags("m", [])
    assert "X052_MODEL_DIR=/models/m" in f


def test_mutate_flags_with_cmd():
    f = push_image.mutate_flags("m", ["python3", "/app/supervise.py"])
    assert f[:6] == ["--cmd", "python3", "--cmd", "/app/supervise.py", "-w", "/app"]
    assert f[6:] == push_image.mutate_flags("m")[6:]  # 環境変数の並びは同じ


def _fake_crane(tmp_path):
    log = tmp_path / "crane_calls.txt"
    sh = tmp_path / "fakecrane"
    sh.write_text(f'#!/bin/sh\necho "$@" >> {log}\n[ "$1" = digest ] && echo sha256:abc\nexit 0\n')
    sh.chmod(0o755)
    return sh, log


def _run_push(tmp_path, monkeypatch, *extra):
    sh, log = _fake_crane(tmp_path)
    layer = tmp_path / "l.tar"
    layer.write_bytes(b"")
    monkeypatch.setenv("X052_IMAGE_REF", "127.0.0.1:5999/x052/t:v1")
    r = subprocess.run([sys.executable, str(TOOLS / "push_image.py"), "--local", "-f", str(layer), "--model-name", "m",
                        "--crane", str(sh), *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [ln for ln in log.read_text().splitlines() if ln.startswith("mutate")][0]


def test_push_image_cmd_flag_reaches_crane_mutate_and_default_is_unchanged(tmp_path, monkeypatch):
    default = _run_push(tmp_path, monkeypatch)
    assert "--cmd sleep --cmd infinity -w /app" in default and "supervise" not in default
    (tmp_path / "crane_calls.txt").unlink()
    custom = _run_push(tmp_path, monkeypatch, "--cmd", "python3", "--cmd", "/app/supervise.py")
    assert "--cmd python3 --cmd /app/supervise.py -w /app" in custom and "sleep" not in custom


class _FakeCrane:
    """crane の代わりに、決まった目録と config を返す（verify の起動の命令の照合だけを試す）。"""
    CMD = ["python3", "/app/supervise.py"]

    def __init__(self, crane, insecure, secret, platform):
        pass

    def json(self, sub, ref):
        base_l = {"digest": "sha256:b", "size": 1}
        own_l = {"digest": "sha256:o", "size": 5}
        if ref == "base":
            return {"layers": [base_l]} if sub == "manifest" else {"rootfs": {"diff_ids": ["sha256:B"]}, "config": {}}
        if sub == "manifest":
            return {"layers": [base_l, own_l]}
        return {"os": "linux", "architecture": "amd64", "rootfs": {"diff_ids": ["sha256:B", "sha256:O"]},
                "config": {"Cmd": self.CMD, "WorkingDir": "/app", "Env": [
                    "PATH=/usr/bin", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "X052_BACKEND=transformers",
                    "PYTHONPATH=/opt/x052/site", "X052_MODEL_DIR=/models/m"]}}

    def blob_info(self, ref, expect, limit):
        return 5, "sha256:o"


def test_verify_checks_the_cmd_that_was_passed(monkeypatch):
    monkeypatch.setattr(verify_image, "Crane", _FakeCrane)
    kw = dict(crane="x", insecure=True, layer_tars=[], base_gib=0.01, model_name="m")
    only_cmd = lambda res: [ok for n, ok, _ in res if n.startswith("起動の命令")]  # noqa: E731
    assert only_cmd(verify_image.verify("127.0.0.1:1/x:y", "base", cmd=["python3", "/app/supervise.py"], **kw)) == [True]
    assert only_cmd(verify_image.verify("127.0.0.1:1/x:y", "base", **kw)) == [False]  # 既定（sleep infinity）とは違う
    _FakeCrane.CMD = ["sleep", "infinity"]
    try:
        res = verify_image.verify("127.0.0.1:1/x:y", "base", **kw)
        assert only_cmd(res) == [True] and "起動の命令 sleep infinity" in {n for n, _, _ in res}
        assert only_cmd(verify_image.verify("127.0.0.1:1/x:y", "base", cmd=["python3", "/app/supervise.py"], **kw)) == [False]
    finally:
        _FakeCrane.CMD = ["python3", "/app/supervise.py"]
