#!/usr/bin/env python3
"""箱の起動の命令（CMD）。server.py を子のプロセスで起こし、落ちたら間をおいて起こし直す（既定 5 秒・5 回まで）。

起こし直しの上限に達しても、自分は終わらずに眠り続ける（箱が止まると、後の呼び出しが全部失敗するため）。
環境変数: X052_SUPERVISE_DELAY（秒・既定 5）・X052_SUPERVISE_MAX_RESTARTS（既定 5）・
X052_SUPERVISE_STABLE_S（この秒より長く動いた子は、起こし直しの数を数え直す・既定 120）・
X052_SERVER_CMD（試験用: 子の命令を差し替える。空白で区切る）。
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from mc2ocr.lifecycle import process_lock

HERE = Path(__file__).resolve().parent
_child = None
_stop = False


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), "supervise:", msg, flush=True)


def _on_term(signum, frame):
    global _stop
    _stop = True
    if _child is not None and _child.poll() is None:
        try:
            _child.terminate()
        except OSError:
            pass


def _cleanup_stale() -> None:
    sdir = os.environ.get("X052_STATE_DIR") or "/app/state"
    for n in ("ready",):
        try:
            os.unlink(os.path.join(sdir, n))  # 死んだ常駐の ready を残さない
        except OSError:
            pass


def _supervise() -> int:
    global _child
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    delay = _num("X052_SUPERVISE_DELAY", 5)
    max_restarts = int(_num("X052_SUPERVISE_MAX_RESTARTS", 5))
    stable = _num("X052_SUPERVISE_STABLE_S", 120)
    cmd = shlex.split(os.environ.get("X052_SERVER_CMD", "")) or [sys.executable, str(HERE / "server.py")]
    restarts = 0
    while not _stop:
        t0 = time.monotonic()
        log(f"starting server (restart {restarts}/{max_restarts}): {cmd[-1]}")
        try:
            _child = subprocess.Popen(cmd)
            rc = _child.wait()
        except Exception as e:  # noqa: BLE001
            log(f"could not start server: {type(e).__name__}: {e}")
            rc = -1
        _cleanup_stale()
        if _stop:
            break
        if time.monotonic() - t0 >= stable:
            restarts = 0
        log(f"server exited with {rc}")
        if restarts >= max_restarts:
            log("restart limit reached; staying alive without a server")
            break
        restarts += 1
        time.sleep(delay)
    while not _stop:  # 眠り続ける（SIGTERM で終わる）
        time.sleep(1)
    return 0


def main(lock_fd=None) -> int:
    sdir = os.environ.get("X052_STATE_DIR") or "/app/state"
    with process_lock(sdir, "supervise", lock_fd) as lock:
        if lock is None:
            return 0
        (Path(sdir) / "supervise.pid").write_text(str(os.getpid()))
        return _supervise()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-fd", type=int)  # app.py passes its already-held lock
    raise SystemExit(main(parser.parse_args().lock_fd))
