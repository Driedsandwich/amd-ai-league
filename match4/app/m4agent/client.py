"""Bounded newline-delimited JSON over a resident Unix socket."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

MAX_MESSAGE = 256 * 1024


def socket_path():
    return os.environ.get("X052_SOCKET", "/tmp/x052_m4.sock")


def receive(conn, deadline):
    data = bytearray()
    while len(data) < MAX_MESSAGE:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("socket deadline")
        conn.settimeout(remaining)
        part = conn.recv(min(65536, MAX_MESSAGE - len(data)))
        if not part:
            break
        data.extend(part)
        if b"\n" in part:
            break
    if len(data) >= MAX_MESSAGE:
        raise ValueError("oversized message")
    return json.loads(bytes(data).split(b"\n", 1)[0])


def call(payload, timeout):
    if timeout <= 0:
        return None
    end = time.monotonic() + timeout
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(socket_path())
            conn.sendall((json.dumps(payload) + "\n").encode())
            response = receive(conn, end)
            return response if isinstance(response, dict) and response.get("ok") else None
    except (OSError, ValueError, TypeError):
        return None


def ensure_server():
    # Existing sockets may be busy; never start another model just because ping times out.
    if os.path.exists(socket_path()):
        return
    script = Path(__file__).resolve().parents[1] / "supervise.py"
    try:
        subprocess.Popen([sys.executable, str(script)], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True)
    except OSError:
        pass


def wait_ready(deadline):
    while time.monotonic() < deadline:
        if call({"op": "ping"}, min(0.25, deadline - time.monotonic())):
            return True
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return False
