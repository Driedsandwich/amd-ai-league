#!/usr/bin/env python3
"""Match 2 pattern: initial launch, up to five restarts, then stay alive."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from m4agent.client import socket_path


def main():
    path = socket_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path + ".supervisor.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        child = None
        stopped = False

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child is not None and child.poll() is None:
                child.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        atomic_pid = Path(path + ".supervisor.pid")
        atomic_pid.write_text(str(os.getpid()))
        try:
            for attempt in range(6):
                if stopped:
                    break
                child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("server.py"))])
                while child.poll() is None and not stopped:
                    time.sleep(0.1)
                if stopped:
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                print(f"server exited {child.returncode}; launch {attempt + 1}/6", file=sys.stderr, flush=True)
                if attempt < 5:
                    for _ in range(50):
                        if stopped:
                            break
                        time.sleep(0.1)
            while not stopped:
                time.sleep(0.2)
        finally:
            atomic_pid.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
