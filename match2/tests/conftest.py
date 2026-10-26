"""手元（Mac・模型なし）の試験の部品。常駐は StubBackend で起こす。"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from PIL import Image

BOX = Path(__file__).resolve().parent.parent
APP = BOX / "app"
TOOLS = BOX / "tools"
sys.path.insert(0, str(APP))


def wait_for(cond, timeout=15.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return None


@pytest.fixture
def short_dir():
    """unix socket のパスが長すぎない短い場所（pytest の tmp_path は長い）。"""
    d = Path(tempfile.mkdtemp(prefix="mc2t_", dir="/tmp"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def env(short_dir):
    e = dict(os.environ)
    e.update(X052_BACKEND="stub", X052_STATE_DIR=str(short_dir / "state"), X052_OUTPUT_DIR=str(short_dir / "out"),
             X052_IMAGE_BUDGET="6", X052_READY_MARGIN="1")
    for k in list(e):
        if k.startswith("X052_STUB_") or k.startswith("X052_SUPERVISE") or k == "X052_SERVER_CMD":
            del e[k]
    return e


@pytest.fixture
def png(short_dir):
    p = short_dir / "image_01.png"
    Image.new("RGB", (700, 300), (240, 240, 240)).save(p)
    return p


@pytest.fixture
def start_server(env):
    """StubBackend の常駐を起こす。返す物＝(Popen, 状態の置き場)。終わりに止める。"""
    procs = []

    def _start(extra=None, wait_ready=True):
        e = dict(env)
        e.update(extra or {})
        p = subprocess.Popen([sys.executable, str(APP / "server.py")], env=e, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(p)
        sdir = Path(e["X052_STATE_DIR"])
        if wait_ready:
            assert wait_for(lambda: (sdir / "ready").exists() or p.poll() is not None, 20)
        return p, sdir

    yield _start
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        if p.stdout:
            p.stdout.close()


def run_app(image, env, timeout=40):
    t = time.monotonic()
    r = subprocess.run([sys.executable, str(APP / "app.py"), "--input-image", str(image)], env=env,
                       capture_output=True, text=True, timeout=timeout)
    return r, time.monotonic() - t


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
