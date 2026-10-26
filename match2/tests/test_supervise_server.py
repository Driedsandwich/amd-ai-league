"""supervise.py と server.py: 読み込み失敗の記録・状態の置き場・死んだ server の起こし直し・自分は生き続ける。"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from conftest import APP, read_json, run_app, wait_for


def start_supervise(env, extra=None):
    e = dict(env)
    e.update(extra or {})
    return subprocess.Popen([sys.executable, str(APP / "supervise.py")], env=e, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    if p.stdout:
        p.stdout.close()


def test_server_status_files(start_server):
    p, sdir = start_server()
    st = read_json(sdir / "server_status.json")
    assert st["state"] == "ready" and st["pid"] == p.pid
    assert {"python", "pythonpath", "torch", "numpy", "PIL"} <= set(st["runtime_env"])  # mc3 の server.py と同じ形
    assert (sdir / "ready").exists() and (sdir / "server.pid").read_text().strip() == str(p.pid)
    assert Path((sdir / "sock_path").read_text().strip()).exists()


def test_load_failure_is_recorded_and_no_ready(start_server):
    p, sdir = start_server({"X052_STUB_FAIL_LOAD": "1"})
    assert wait_for(lambda: p.poll() is not None, 10)
    assert p.returncode == 1
    st = read_json(sdir / "server_status.json")
    assert st["state"] == "load_failed" and "failed on purpose" in st["error"]
    assert not (sdir / "ready").exists()


def test_supervise_restarts_a_killed_server_and_stays_alive(env, png):
    env.update(X052_SUPERVISE_DELAY="0.2", X052_STUB_ANSWER="AGAIN1")
    sup = start_supervise(env)
    try:
        sdir = Path(env["X052_STATE_DIR"])
        assert wait_for(lambda: (sdir / "ready").exists(), 20)
        pid1 = int((sdir / "server.pid").read_text())
        os.kill(pid1, signal.SIGKILL)
        def new_pid():
            try:
                pid2 = int((sdir / "server.pid").read_text())
            except (OSError, ValueError):
                return None
            return pid2 if pid2 != pid1 and (sdir / "ready").exists() else None
        pid2 = wait_for(new_pid, 20)
        assert pid2 and pid2 != pid1, "server が起こし直された"
        assert sup.poll() is None, "supervise 自身は生きている"
        r, _ = run_app(png, env)
        assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": "AGAIN1"}
    finally:
        stop(sup)


def test_supervise_gives_up_after_limit_but_keeps_sleeping(env, short_dir):
    log = short_dir / "runs.txt"
    script = short_dir / "die.py"
    script.write_text(f"import sys\nopen({str(log)!r}, 'a').write('x')\nsys.exit(3)\n")
    sup = start_supervise(env, {"X052_SERVER_CMD": f"{sys.executable} {script}", "X052_SUPERVISE_DELAY": "0.1",
                                "X052_SUPERVISE_MAX_RESTARTS": "2"})
    try:
        assert wait_for(lambda: log.exists() and len(log.read_text()) >= 3, 15)  # 最初の 1 回＋起こし直し 2 回
        time.sleep(1.5)
        assert len(log.read_text()) == 3, "上限を超えて起こし直さない"
        assert sup.poll() is None, "上限に達しても supervise は終わらない"
    finally:
        stop(sup)
    assert sup.returncode == 0  # SIGTERM で静かに終わる


def test_supervise_removes_stale_ready_when_server_dies(env, short_dir):
    sdir = Path(env["X052_STATE_DIR"])
    sdir.mkdir(parents=True)
    (sdir / "ready").write_text("1")
    script = short_dir / "die.py"
    script.write_text("import sys\nsys.exit(1)\n")
    sup = start_supervise(env, {"X052_SERVER_CMD": f"{sys.executable} {script}", "X052_SUPERVISE_DELAY": "0.1",
                                "X052_SUPERVISE_MAX_RESTARTS": "0"})
    try:
        assert wait_for(lambda: not (sdir / "ready").exists(), 10)
    finally:
        stop(sup)
