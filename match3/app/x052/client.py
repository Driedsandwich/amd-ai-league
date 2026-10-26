"""問いごとの app.py から常駐へ問いを渡す。常駐が無い・遅いときは None。

受け取りは time.monotonic() の 1 つの絶対の期限で数える（少しずつ届く応答でも期限は延びない）。
"""
from __future__ import annotations

import json
import os
import socket
import time
from typing import Optional


def socket_path(index_dir: str) -> str:
    rec = os.path.join(index_dir, "sock_path")
    try:
        p = open(rec).read().strip()
        if p:
            return p
    except OSError:
        pass
    return os.path.join(index_dir, "x052.sock")


def _call(index_dir: str, payload: dict, timeout: float) -> Optional[dict]:
    deadline = time.monotonic() + timeout
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        rem = deadline - time.monotonic()
        if rem <= 0:
            return None
        s.settimeout(rem)
        s.connect(socket_path(index_dir))
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


def ask(index_dir: str, query: str, timeout: float = 25.0) -> Optional[dict]:
    """timeout 秒以内に答えが無ければ None。常駐へは壁時計の期限も渡し、古い問いを常駐が捨てられるようにする。"""
    return _call(index_dir, {"op": "query", "query": query, "expires_wall": time.time() + timeout}, timeout)


def ping(index_dir: str, timeout: float = 3.0) -> Optional[dict]:
    r = _call(index_dir, {"op": "ping"}, timeout)
    return r if r and r.get("ok") else None


def shutdown(index_dir: str, timeout: float = 5.0) -> Optional[dict]:
    return _call(index_dir, {"op": "shutdown"}, timeout)
