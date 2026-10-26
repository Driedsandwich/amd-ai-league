#!/usr/bin/env python3
"""Load and warm up once, then serve one generation per Unix connection."""
import fcntl
import json
import os
import socket
import time
import traceback
from pathlib import Path

from m4agent.backend import make_backend
from m4agent.client import receive, socket_path
from m4agent.common import atomic_json


def serve():
    path = socket_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path + ".server.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        try:
            backend = make_backend()
            backend.load()
            backend.warmup()
        except Exception as exc:
            atomic_json(path + ".status.json", {"state": "load_failed", "error": str(exc)})
            traceback.print_exc()
            return 1
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            os.chmod(path, 0o600)
            listener.listen(8)
            atomic_json(path + ".status.json", {"state": "ready", "pid": os.getpid()})
            try:
                while True:
                    conn, _ = listener.accept()
                    with conn:
                        try:
                            req = receive(conn, time.monotonic() + 2)
                            op = req.get("op")
                            if op == "ping":
                                reply = {"ok": True, "pid": os.getpid()}
                            elif op == "generate":
                                seconds = min(24.0, max(0.0, float(req.get("max_time", 1))))
                                # Absolute monotonic deadline prevents stale queued generation.
                                seconds = min(seconds, float(req.get("deadline", time.monotonic() + seconds))
                                              - time.monotonic())
                                text = backend.generate(str(req.get("prompt", "")),
                                                        min(64, max(1, int(req.get("max_new", 64)))),
                                                        seconds) if seconds > 0 else ""
                                reply = {"ok": True, "text": text}
                            else:
                                reply = {"ok": False, "error": "unknown op"}
                            conn.settimeout(0.5)
                            conn.sendall((json.dumps(reply) + "\n").encode())
                        except Exception:
                            traceback.print_exc()
            finally:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    raise SystemExit(serve())
