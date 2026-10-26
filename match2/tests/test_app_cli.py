"""入口 app.py の約束: 出力の名前・JSON の形・置き換えの書き方・常駐が無い／遅いときも締切の内に空で 0 で終わる。"""
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

import app as appmod
from conftest import APP, read_json, run_app, wait_for


@pytest.mark.parametrize("name,out", [
    ("image_01.png", "image_01_output.json"),
    ("image_02.JPG", "image_02_output.json"),
    ("image_03.jpeg", "image_03_output.json"),
    ("image_04.tif", "image_04_output.json"),
    ("image_05.TIFF", "image_05_output.json"),
    ("a.b.png", "a.b_output.json"),
])
def test_output_name(name, out):
    assert appmod.output_name("/app/input/" + name) == out


def test_write_output_shape_and_atomic_replace(short_dir, monkeypatch):
    monkeypatch.setenv("X052_OUTPUT_DIR", str(short_dir / "o"))
    calls = []
    real = os.replace
    monkeypatch.setattr(appmod.os, "replace", lambda s, d: (calls.append((Path(s).name, Path(d).name)), real(s, d))[1])
    appmod.write_output("/x/image_01.PNG", "京 A·12345")
    assert calls == [(".image_01_output.json.tmp", "image_01_output.json")]
    obj = read_json(short_dir / "o" / "image_01_output.json")
    assert obj == {"text": "京 A·12345"}  # ensure_ascii=False でも読み戻せる
    assert sorted(p.name for p in (short_dir / "o").iterdir()) == ["image_01_output.json"]  # 一時のファイルは残らない
    assert (short_dir / "o" / "image_01_output.json").read_text(encoding="utf-8").count("京") == 1


def test_no_server_writes_empty_inside_budget_and_exits_zero(env, png):
    env.update(X052_IMAGE_BUDGET="2.5", X052_READY_MARGIN="0.5")
    r, dt = run_app(png, env)
    assert r.returncode == 0, r.stderr
    assert dt < 4.0, dt  # 締切（2.5 秒）の内
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}


def test_missing_image_still_writes_empty(env, short_dir):
    env.update(X052_IMAGE_BUDGET="1.5", X052_READY_MARGIN="0.2")
    r, dt = run_app(short_dir / "nope.png", env)
    assert r.returncode == 0
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "nope_output.json") == {"text": ""}


def test_unwritable_output_dir_exits_zero(env, png, short_dir):
    blocker = short_dir / "afile"
    blocker.write_text("x")
    env.update(X052_OUTPUT_DIR=str(blocker / "sub"), X052_IMAGE_BUDGET="1.2", X052_READY_MARGIN="0.1")
    r, _ = run_app(png, env)
    assert r.returncode == 0  # 書けなくても例外で落ちない


def test_missing_argument_exits_zero(env):
    import subprocess
    r = subprocess.run([sys.executable, str(APP / "app.py")], env=env, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0


@pytest.mark.parametrize("name", ["image_01.png", "image_02.JPG", "image_03.jpeg", "image_04.tif", "image_05.TIFF"])
def test_with_stub_server_answer_returned_and_cleaned(env, start_server, short_dir, name):
    p, sdir = start_server({"X052_STUB_ANSWER": 'Text: "8KQM417."'})
    img = short_dir / name
    Image.new("RGB", (800, 300), (255, 255, 255)).save(img)
    r, dt = run_app(img, env)
    assert r.returncode == 0, r.stderr
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / (Path(name).stem + "_output.json")) == {"text": "8KQM417"}
    assert dt < 6


def test_answer_from_file_is_reread_each_call(env, start_server, png, short_dir):
    f = short_dir / "ans.txt"
    f.write_text("first\n")
    p, sdir = start_server({"X052_STUB_ANSWER_FILE": str(f)})
    r, _ = run_app(png, env)
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": "first"}
    f.write_text("second line\nnext")
    r, _ = run_app(png, env)
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": "second line next"}


def test_slow_server_gets_empty_answer_inside_deadline_and_survives(env, start_server, png):
    p, sdir = start_server({"X052_STUB_SLEEP": "30"})
    env.update(X052_IMAGE_BUDGET="4")
    r, dt = run_app(png, env)
    assert r.returncode == 0
    assert dt < 5.5, dt  # 締切（4 秒）を大きく超えない
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}
    from mc2ocr import client
    assert wait_for(lambda: client.ping(str(sdir)), 10), "常駐は生きている"
    assert p.poll() is None


def test_waits_for_ready_then_answers(env, start_server, png):
    """最初の呼び出しが模型の読み込み中に来ても、締切の内に ready になれば答えを返す。"""
    env.update(X052_IMAGE_BUDGET="9", X052_READY_MARGIN="1")
    p, sdir = start_server({"X052_STUB_LOAD_SLEEP": "2", "X052_STUB_ANSWER": "WAITED"}, wait_ready=False)
    r, dt = run_app(png, env)
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": "WAITED"}
    assert dt >= 1.5


def test_alarm_last_resort_writes_empty(env, start_server, png):
    """常駐が締切を無視して固まっても、最後の砦（アラーム）が空の答えを書いて 0 で終わる。"""
    p, sdir = start_server()
    os.kill(p.pid, signal.SIGSTOP)  # SIGSTOP: 常駐が固まる（接続は受け付けられたまま返事が来ない）
    try:
        env.update(X052_IMAGE_BUDGET="10", X052_ALARM_S="2", X052_READY_MARGIN="1")  # 締切より先にアラームが鳴る
        r, dt = run_app(png, env)
        assert r.returncode == 0 and 1.5 < dt < 4.0, dt  # 締切（10 秒）ではなくアラーム（2 秒）で終わった
        assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}
    finally:
        os.kill(p.pid, signal.SIGCONT)
