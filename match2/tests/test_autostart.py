"""CMD override: detached, single-instance startup with the stub backend."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from conftest import APP, read_json, run_app, wait_for
from mc2ocr import client
from mc2ocr.lifecycle import process_lock
from test_supervise_server import start_supervise, stop


@pytest.fixture
def autostart(env):
    env.pop("X052_TEST_SUPERVISE_PYTHON", None)
    sdir = Path(env["X052_STATE_DIR"])
    yield sdir
    try:
        pid = int((sdir / "supervise.pid").read_text())
    except FileNotFoundError:
        return
    # The detached process group includes its server; never leave it behind.
    try:
        os.killpg(pid, signal.SIGTERM)

        def stopped():
            with process_lock(str(sdir), "supervise") as lock:
                return lock is not None

        wait_for(stopped, 5)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_cold_start_survives_first_deadline_and_later_call_answers(env, png, autostart):
    sdir = autostart
    sdir.mkdir(parents=True)
    (sdir / "supervise.log").write_text("previous log\n")
    env.update(X052_STUB_LOAD_SLEEP="1.5", X052_STUB_ANSWER="LATER42",
               X052_IMAGE_BUDGET="1", X052_READY_MARGIN="0.5")
    first, elapsed = run_app(png, env)
    assert first.returncode == 0, first.stderr
    assert elapsed < 1
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}
    assert wait_for(lambda: client.is_ready(str(sdir)), 20)
    pid = int((sdir / "supervise.pid").read_text())
    assert os.getsid(pid) == pid
    assert os.readlink(f"/proc/{pid}/fd/0") == "/dev/null"
    env["X052_IMAGE_BUDGET"] = "6"
    later, _ = run_app(png, env)
    assert later.returncode == 0, later.stderr
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": "LATER42"}
    assert int((sdir / "supervise.pid").read_text()) == pid
    log = (sdir / "supervise.log").read_text()
    assert log.startswith("previous log\n")
    assert log.count("starting server") == 1


def test_concurrent_apps_start_only_one_supervisor(env, png, short_dir, autostart):
    env.update(X052_STUB_LOAD_SLEEP="1", X052_STUB_ANSWER="TOGETHER42", X052_IMAGE_BUDGET="12")
    second_image = short_dir / "image_02.png"
    second_image.write_bytes(png.read_bytes())
    gate_read, gate_write = os.pipe()
    procs = []
    try:
        for image in (png, second_image):
            # Both processes wait at the same gate before executing app.py.
            code = ("import os, sys; os.read(int(sys.argv[1]), 1); "
                    "os.close(int(sys.argv[1])); os.execv(sys.executable, [sys.executable] + sys.argv[2:])")
            procs.append(subprocess.Popen(
                [sys.executable, "-c", code, str(gate_read), str(APP / "app.py"), "--input-image", str(image)],
                env=env, pass_fds=(gate_read,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        os.write(gate_write, b"xx")
        for proc in procs:
            _, stderr = proc.communicate(timeout=15)
            assert proc.returncode == 0, stderr
        for image in (png, second_image):
            assert read_json(Path(env["X052_OUTPUT_DIR"]) / f"{image.stem}_output.json") == {"text": "TOGETHER42"}
        assert (autostart / "supervise.log").read_text().count("starting server") == 1
        # The one supervisor keeps the inherited lock after both clients exit.
        with process_lock(str(autostart), "supervise") as lock:
            assert lock is None
    finally:
        os.close(gate_read)
        os.close(gate_write)
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.communicate()


def test_spawn_failure_writes_empty_within_budget(env, png, autostart):
    env.update(X052_TEST_SUPERVISE_PYTHON="/nonexistent/mc2-python",
               X052_IMAGE_BUDGET="2", X052_READY_MARGIN="0.5")
    result, elapsed = run_app(png, env, timeout=3)
    assert result.returncode == 0, result.stderr
    assert elapsed < 2
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}
    assert (autostart / "supervise.log").exists()  # startup was attempted
    assert not (autostart / "supervise.pid").exists()
    with process_lock(str(autostart), "supervise") as lock:
        assert lock is not None  # a failed exec does not leave startup locked


@pytest.mark.parametrize("loading", [False, True])
def test_existing_server_does_not_spawn(env, png, start_server, loading):
    proc, sdir = start_server({"X052_STUB_LOAD_SLEEP": "2" if loading else "0",
                               "X052_STUB_ANSWER": "EXISTING42"}, wait_ready=not loading)
    if loading:
        def has_lock():
            with process_lock(str(sdir), "server") as lock:
                return lock is None
        assert wait_for(has_lock)
        assert not client.is_ready(str(sdir))
    result, _ = run_app(png, env)
    assert result.returncode == 0, result.stderr
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": "EXISTING42"}
    assert int((sdir / "server.pid").read_text()) == proc.pid
    assert not (sdir / "supervise.log").exists()
    assert not (sdir / "supervise.pid").exists()


def test_existing_supervisor_without_ready_does_not_spawn(env, png):
    env.update(X052_STUB_FAIL_LOAD="1", X052_SUPERVISE_MAX_RESTARTS="0",
               X052_IMAGE_BUDGET="2", X052_READY_MARGIN="0.5")
    proc = start_supervise(env)
    sdir = Path(env["X052_STATE_DIR"])
    try:
        assert wait_for(lambda: (sdir / "server_status.json").exists(), 20)
        result, elapsed = run_app(png, env)
        assert result.returncode == 0, result.stderr
        assert elapsed < 2
        assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}
        assert proc.poll() is None
        assert not (sdir / "supervise.log").exists()
        assert int((sdir / "supervise.pid").read_text()) == proc.pid
    finally:
        stop(proc)


def test_stalled_spawn_is_bounded_by_existing_deadline(env, png, short_dir):
    # A blocked log open exercises the alarm around startup, before Popen.
    sdir = Path(env["X052_STATE_DIR"])
    sdir.mkdir(parents=True)
    os.mkfifo(sdir / "supervise.log")
    env.update(X052_IMAGE_BUDGET="2", X052_READY_MARGIN="0.5")
    result, elapsed = run_app(png, env, timeout=3)
    assert result.returncode == 0, result.stderr
    assert elapsed < 2
    assert read_json(Path(env["X052_OUTPUT_DIR"]) / "image_01_output.json") == {"text": ""}
