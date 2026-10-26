"""app.py を本物のプロセスで呼ぶ。常駐なしと、StubBackend の常駐ありの両方。"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app" / "app.py"


def run(args, env, timeout=60):
    return subprocess.run([sys.executable, str(APP), *args], env=env, capture_output=True,
                          text=True, timeout=timeout)


def base_env(tmp_path, **extra):
    env = dict(os.environ)
    env.update(X052_INDEX_DIR=str(tmp_path / "index"), X052_OUTPUT_DIR=str(tmp_path / "out"),
               X052_BACKEND="stub", X052_STUB_ANSWER="94")
    env.update(extra)
    return env


def check_shape(path, query_id):
    assert path.name == f"{query_id}_output.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(obj["answer"], str)
    assert isinstance(obj["citations"], list) and all(isinstance(c, str) for c in obj["citations"])
    assert isinstance(obj["confidence"], float)
    return obj


def test_index_then_query_without_server(corpus, tmp_path):
    env = base_env(tmp_path, X052_NO_SERVER="1")
    r = run(["--index", str(corpus)], env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "index" / "chunks.jsonl").exists()
    manifest = json.loads((tmp_path / "index" / "manifest.json").read_text())
    assert any(m.get("reason") == "encrypted" for m in manifest)
    r = run(["--corpus", str(corpus), "--query-id", "query_07", "--query", "What is the max temperature?"], env)
    assert r.returncode == 0, r.stderr
    obj = check_shape(tmp_path / "out" / "query_07_output.json", "query_07")
    assert obj["answer"] == "" and obj["citations"] == []  # 常駐なしは空の答え


def test_query_with_no_index_at_all_still_writes(corpus, tmp_path):
    env = base_env(tmp_path)
    r = run(["--corpus", str(corpus), "--query-id", "query_01", "--query", "q"], env)
    assert r.returncode == 0
    check_shape(tmp_path / "out" / "query_01_output.json", "query_01")


def test_index_and_query_with_stub_server(corpus, tmp_path):
    env = base_env(tmp_path, X052_READY_TIMEOUT="60")
    try:
        r = run(["--index", str(corpus)], env, timeout=90)
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "index" / "ready").exists(), (tmp_path / "index" / "server.log").read_text()
        r = run(["--corpus", str(corpus), "--query-id", "query_03", "--query",
                 "What is the maximum junction temperature of the Widget W9?"], env)
        assert r.returncode == 0, r.stderr
        obj = check_shape(tmp_path / "out" / "query_03_output.json", "query_03")
        assert obj["answer"] == "94"
        assert obj["citations"] == ["specs/widget_datasheet_r2.pdf"]
        # 新しいプロセスで 2 問目も同じ常駐が答える
        r = run(["--corpus", str(corpus), "--query-id", "query_04", "--query", "error code thermal throttle"], env)
        check_shape(tmp_path / "out" / "query_04_output.json", "query_04")
    finally:
        from x052 import client
        client.shutdown(str(tmp_path / "index"))
        pidf = tmp_path / "index" / "server.pid"
        if pidf.exists():
            time.sleep(0.5)
            try:
                os.kill(int(pidf.read_text()), 15)
            except (ProcessLookupError, ValueError):
                pass


def test_no_leftover_tmp_in_output(corpus, tmp_path):
    env = base_env(tmp_path, X052_NO_SERVER="1")
    run(["--corpus", str(corpus), "--query-id", "query_02", "--query", "q"], env)
    names = [p.name for p in (tmp_path / "out").iterdir()]
    assert names == ["query_02_output.json"]
