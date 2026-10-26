"""画像ごとの app.py から常駐へ依頼を渡す（unix socket だけ。外へ出る通信は無い）。

受け取りは time.monotonic() の 1 つの絶対の期限で数える。常駐が無い・遅い・壊れた返事は None。
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from typing import Optional


def default_sock_path(state_dir: str) -> str:
    """unix socket のパスは約 104 字まで。長いときは短い場所に置く（常駐は sock_path にも記録する）。"""
    p = os.path.join(state_dir, "x052.sock")
    if len(p.encode()) < 100:
        return p
    h = hashlib.sha1(os.path.abspath(state_dir).encode()).hexdigest()[:10]
    q = os.path.join(tempfile.gettempdir(), f"x052-{h}.sock")
    if len(q.encode()) >= 100:
        q = f"/tmp/x052-{h}.sock"
    return q


def socket_path(state_dir: str) -> str:
    try:
        with open(os.path.join(state_dir, "sock_path")) as f:
            p = f.read().strip()
        if p:
            return p
    except OSError:
        pass
    return default_sock_path(state_dir)


def is_ready(state_dir: str) -> bool:
    return os.path.exists(os.path.join(state_dir, "ready"))


def _call(state_dir: str, payload: dict, timeout: float) -> Optional[dict]:
    deadline = time.monotonic() + timeout
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        rem = deadline - time.monotonic()
        if rem <= 0:
            return None
        s.settimeout(rem)
        s.connect(socket_path(state_dir))
        s.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        buf = b""
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                return None
            s.settimeout(rem)
            part = s.recv(65536)
            if not part:
                break
            buf += part
            if buf.endswith(b"\n"):
                break
        return json.loads(buf.decode("utf-8"))
    except Exception:  # noqa: BLE001 - 常駐なし・時間切れ・壊れた返事はすべて None
        return None
    finally:
        s.close()


def read_image(state_dir: str, image_path: str, deadline_wall: float) -> Optional[dict]:
    """deadline_wall（UNIX 秒）までに答えが無ければ None。"""
    timeout = deadline_wall - time.time()
    return _call(state_dir, {"image_path": image_path, "deadline": deadline_wall}, timeout)


def ping(state_dir: str, timeout: float = 3.0) -> Optional[dict]:
    r = _call(state_dir, {"op": "ping"}, timeout)
    return r if r and r.get("ok") else None


def shutdown(state_dir: str, timeout: float = 5.0) -> Optional[dict]:
    return _call(state_dir, {"op": "shutdown"}, timeout)
