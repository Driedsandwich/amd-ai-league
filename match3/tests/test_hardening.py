"""監査の直しの試験: 起動の期限・常駐の死・遅い生成・画像の時間切れ・大きいファイル・書き込みの失敗。"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from x052 import client, indexer, parsers
from x052.server import add_image_chunks

APP = Path(__file__).resolve().parent.parent / "app" / "app.py"


def run(args, env, timeout=60):
    return subprocess.run([sys.executable, str(APP), *args], env=env, capture_output=True, text=True,
                          timeout=timeout)


def base_env(tmp_path, **extra):
    env = dict(os.environ)
    env.update(X052_INDEX_DIR=str(tmp_path / "index"), X052_OUTPUT_DIR=str(tmp_path / "out"),
               X052_BACKEND="stub", X052_STUB_ANSWER="94", X052_READY_TIMEOUT="60")
    env.update(extra)
    return env


def read_out(tmp_path, qid):
    obj = json.loads((tmp_path / "out" / f"{qid}_output.json").read_text(encoding="utf-8"))
    assert isinstance(obj["answer"], str)
    assert isinstance(obj["citations"], list) and all(isinstance(c, str) for c in obj["citations"])
    return obj


def kill_server(tmp_path):
    client.shutdown(str(tmp_path / "index"), timeout=2)
    pidf = tmp_path / "index" / "server.pid"
    if pidf.exists():
        try:
            os.kill(int(pidf.read_text()), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    st = tmp_path / "index" / "startup_status.json"
    if st.exists():
        try:
            os.kill(int(json.loads(st.read_text())["server_pid"]), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


Q = "What is the maximum junction temperature of the Widget W9?"


# ---- 指摘 1・3・11: 常駐の失敗 ----
def test_model_load_failure_gives_empty_json_and_status(corpus, tmp_path):
    env = base_env(tmp_path, X052_STUB_FAIL_LOAD="1")
    t0 = time.monotonic()
    r = run(["--index", str(corpus)], env)
    assert r.returncode == 0 and time.monotonic() - t0 < 30
    st = json.loads((tmp_path / "index" / "startup_status.json").read_text())
    assert st["state"] == "server_exited"
    assert json.loads((tmp_path / "index" / "server_status.json").read_text())["state"] == "load_failed"
    r = run(["--corpus", str(corpus), "--query-id", "query_01", "--query", Q], env)
    assert r.returncode == 0
    assert read_out(tmp_path, "query_01") == {"answer": "", "citations": [], "confidence": 0.0}


def test_startup_deadline_when_model_is_too_slow(corpus, tmp_path):
    env = base_env(tmp_path, X052_STUB_LOAD_SLEEP="30", X052_INDEX_DEADLINE="4")
    try:
        t0 = time.monotonic()
        r = run(["--index", str(corpus)], env)
        assert r.returncode == 0 and time.monotonic() - t0 < 10  # 期限で戻る（終了コードは 0）
        st = json.loads((tmp_path / "index" / "startup_status.json").read_text())
        assert st["state"] == "timeout" and not (tmp_path / "index" / "ready").exists()
    finally:
        kill_server(tmp_path)


def test_ready_state_is_recorded_after_ping(corpus, tmp_path):
    env = base_env(tmp_path)
    try:
        assert run(["--index", str(corpus)], env).returncode == 0
        st = json.loads((tmp_path / "index" / "startup_status.json").read_text())
        assert st["state"] == "ready" and st["files_indexed"] >= 6
        assert client.ping(str(tmp_path / "index")) is not None
    finally:
        kill_server(tmp_path)


def test_server_dies_midway_later_queries_still_valid(corpus, tmp_path):
    env = base_env(tmp_path)
    try:
        run(["--index", str(corpus)], env)
        run(["--corpus", str(corpus), "--query-id", "query_01", "--query", Q], env)
        assert read_out(tmp_path, "query_01")["answer"] == "94"
        os.kill(int((tmp_path / "index" / "server.pid").read_text()), signal.SIGKILL)
        time.sleep(0.3)
        for qid in ("query_02", "query_03"):
            t0 = time.monotonic()
            assert run(["--corpus", str(corpus), "--query-id", qid, "--query", Q], env).returncode == 0
            assert read_out(tmp_path, qid) == {"answer": "", "citations": [], "confidence": 0.0}
            assert time.monotonic() - t0 < 5  # 常駐が無ければ待たずに空を書く
    finally:
        kill_server(tmp_path)


# ---- 指摘 3: 遅い 1 問（予算を縮めて回す。既定値の計算は下の単体で）----
def test_slow_generation_is_cut_by_max_time_within_budget(corpus, tmp_path):
    env = base_env(tmp_path, X052_STUB_SLEEP="100", X052_QUERY_BUDGET="8", X052_QUERY_MARGIN="2",
                   X052_ALARM_S="7")
    try:
        run(["--index", str(corpus)], env)
        t0 = time.monotonic()
        r = run(["--corpus", str(corpus), "--query-id", "query_01", "--query", Q], env)
        dt = time.monotonic() - t0
        assert r.returncode == 0 and dt < 8, dt
        assert read_out(tmp_path, "query_01")["answer"] == ""
    finally:
        kill_server(tmp_path)


def test_hung_generation_last_resort_alarm_writes_empty_json(corpus, tmp_path):
    """模型が max_time を無視して固まっても、始まりから alarm 秒で空の JSON を書いて終わる。"""
    env = base_env(tmp_path, X052_STUB_HANG="60", X052_QUERY_BUDGET="100", X052_QUERY_MARGIN="0",
                   X052_ALARM_S="3")
    try:
        run(["--index", str(corpus)], env)
        t0 = time.monotonic()
        r = run(["--corpus", str(corpus), "--query-id", "query_01", "--query", Q], env, timeout=30)
        dt = time.monotonic() - t0
        assert r.returncode == 0 and 2.5 < dt < 8, dt
        assert read_out(tmp_path, "query_01") == {"answer": "", "citations": [], "confidence": 0.0}
    finally:
        kill_server(tmp_path)


def test_default_time_constants():
    src = APP.read_text(encoding="utf-8")
    assert '_envf("X052_QUERY_BUDGET", 30)' in src and '_envf("X052_QUERY_MARGIN", 4)' in src
    assert '_envf("X052_ALARM_S", 28)' in src and '_envf("X052_INDEX_DEADLINE", 480)' in src
    assert '_envf("X052_PARSE_BUDGET", 240)' in src


def test_client_deadline_is_absolute(tmp_path):
    t0 = time.monotonic()
    assert client.ask(str(tmp_path), "q", timeout=0.0) is None
    assert time.monotonic() - t0 < 0.5


def test_write_failure_falls_back_to_plain_write(tmp_path, monkeypatch):
    import app
    monkeypatch.setenv("X052_OUTPUT_DIR", str(tmp_path / "o"))
    monkeypatch.setattr(app, "write_output", lambda *a, **k: (_ for _ in ()).throw(OSError("replace failed")))
    assert app.write_output_safely("query_09", "x", ["a.txt"], 0.5) is True
    obj = json.loads((tmp_path / "o" / "query_09_output.json").read_text())
    assert obj == {"answer": "x", "citations": ["a.txt"], "confidence": 0.5}


# ---- 指摘 2: 画像 ----
def test_image_timeout_does_not_skip_remaining_images(tmp_path, monkeypatch):
    monkeypatch.setenv("X052_STUB_VISION", "1")
    monkeypatch.setenv("X052_STUB_IMAGE_SLEEP", "3")
    from x052.backend import StubBackend
    idx = tmp_path / "idx"
    idx.mkdir()
    names = ["a_slow.png", "b_fast.png", "c_slow.png", "d_fast.png"]
    (idx / "manifest.json").write_text(json.dumps([{"path": n, "status": "image"} for n in names]))
    chunks = []
    res = add_image_chunks(str(idx), str(tmp_path), StubBackend(), chunks, each=0.3, total=30)
    assert [r["status"] for r in res] == ["timeout", "ok", "timeout", "ok"]
    assert {c["path"] for c in chunks} == {"b_fast.png", "d_fast.png"}  # 時間切れの後も次の画像を続ける


def test_image_total_budget_skips_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("X052_STUB_VISION", "1")
    monkeypatch.setenv("X052_STUB_IMAGE_SLEEP", "3")
    from x052.backend import StubBackend
    idx = tmp_path / "idx"
    idx.mkdir()
    names = ["a_slow.png", "b_fast.png", "c_fast.png"]
    (idx / "manifest.json").write_text(json.dumps([{"path": n, "status": "image"} for n in names]))
    res = add_image_chunks(str(idx), str(tmp_path), StubBackend(), [], each=5, total=1.5)
    assert res[0]["status"] == "timeout" and [r["status"] for r in res[1:]] == ["skipped_time", "skipped_time"]


# ---- 指摘 10・1: 大きいファイルと文の取り出しの予算 ----
def test_default_limits():
    assert parsers.TEXT_MAX_BYTES == 200 * 1024 * 1024 and parsers.XLSX_MAX_ROWS == 1_000_000


def test_text_just_after_limit_is_flagged_truncated(tmp_path, monkeypatch):
    c = tmp_path / "c"
    c.mkdir()
    (c / "big.log").write_text("filler line\n" * 300 + "SECRET=ZEBRA42\n", encoding="utf-8")
    (c / "small.log").write_text("filler line\nSECRET=ZEBRA42\n", encoding="utf-8")
    monkeypatch.setattr(parsers, "TEXT_MAX_BYTES", 300 * len("filler line\n"))
    chunks, manifest = indexer.walk_corpus(str(c))
    m = {x["path"]: x for x in manifest}
    assert m["big.log"]["truncated"] is True and m["big.log"]["truncated_reason"].startswith("bytes>")
    assert not any("ZEBRA42" in x["text"] for x in chunks if x["path"] == "big.log")  # 上限の直後の答えは読めない
    assert "truncated" not in m["small.log"] and any("ZEBRA42" in x["text"] for x in chunks)
    # 上限を上げれば読める（既定の 200MiB なら読める大きさ）
    monkeypatch.setattr(parsers, "TEXT_MAX_BYTES", 200 * 1024 * 1024)
    chunks, manifest = indexer.walk_corpus(str(c))
    assert any("ZEBRA42" in x["text"] for x in chunks if x["path"] == "big.log")
    assert not [x for x in manifest if x.get("truncated")]


def test_csv_and_xlsx_limits_are_flagged(tmp_path, monkeypatch):
    import openpyxl
    c = tmp_path / "c"
    c.mkdir()
    (c / "t.csv").write_text("a,b\n" + "".join(f"{i},v{i}\n" for i in range(500)) + "9999,ZEBRA42\n")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["k", "v"])
    for i in range(10):
        ws.append([i, f"v{i}"])
    ws.append([99, "ZEBRA42"])
    wb.save(str(c / "t.xlsx"))
    monkeypatch.setattr(parsers, "TEXT_MAX_BYTES", 1000)
    monkeypatch.setattr(parsers, "XLSX_MAX_ROWS", 5)
    chunks, manifest = indexer.walk_corpus(str(c))
    m = {x["path"]: x for x in manifest}
    assert m["t.csv"]["truncated"] and m["t.xlsx"]["truncated"]
    assert not any("ZEBRA42" in x["text"] for x in chunks)


def test_parse_budget_marks_skipped_time_and_truncated(corpus, tmp_path):
    chunks, manifest = indexer.walk_corpus(str(corpus), parse_budget=0)
    files = [m for m in manifest if m["kind"] != "dir"]
    assert files and all(m["reason"] == "skipped_time" for m in files)
    assert chunks == []
    big = "line of text\n" * 60000
    segs = parsers.chunk_lines(big, deadline=time.monotonic() - 1)
    assert segs.truncated and segs.truncated_reason == "time"
