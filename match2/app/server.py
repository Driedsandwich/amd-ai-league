#!/usr/bin/env python3
"""常駐。起動の時に模型を 1 回だけ読み込み、unix socket で 1 件ずつ画像の依頼を受ける。

状態の置き場（X052_STATE_DIR・既定 /app/state）に書く物:
  server_status.json  {"state": "ready"|"load_failed", ..., "runtime_env": {...}}
  ready               模型を読み終えて待ち受けを始めた印
  x052.sock / sock_path / server.pid

依頼＝{"image_path": ..., "deadline": <UNIX 秒>}。返事＝{"text": ...}。締切を過ぎそうなら生成を切る（空を返す）。
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc2ocr import client, postprocess  # noqa: E402
from mc2ocr.backend import make_backend  # noqa: E402
from mc2ocr.lifecycle import process_lock  # noqa: E402
from mc2ocr.prompt import MAX_NEW_TOKENS, build_prompt  # noqa: E402


def state_dir() -> str:
    return os.environ.get("X052_STATE_DIR") or "/app/state"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def runtime_env() -> dict:
    """本番の像で動いた torch・numpy・PIL の版と置き場所と GPU を残す（依存の層が土台の ROCm 版を影にしていないかを後から読むため）。"""
    env = {"python": sys.version.split()[0], "pythonpath": os.environ.get("PYTHONPATH", "")}
    for name in ("torch", "numpy", "PIL"):
        try:
            mod = __import__(name)
            env[name] = {"version": getattr(mod, "__version__", "?"), "file": getattr(mod, "__file__", "?")}
        except BaseException as e:  # noqa: BLE001 - 試験の手元では torch が無いことがある
            env[name] = {"error": f"{type(e).__name__}: {e}"}
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["hip"] = getattr(torch.version, "hip", None)
        env["device"] = torch.cuda.get_device_name(0) if env["cuda_available"] else None
    except BaseException as e:  # noqa: BLE001
        env["cuda_error"] = f"{type(e).__name__}: {e}"
    return env


def _margin() -> float:
    try:
        return float(os.environ.get("X052_SERVER_MARGIN") or 1.0)
    except ValueError:
        return 1.0


def read_one(backend, image_path: str, deadline_wall: Optional[float]) -> str:
    """画像 1 枚を読んで整えた文字を返す。締切（UNIX 秒）の margin 秒前までに生成を切る。"""
    from mc2ocr import imaging  # acquire the lifetime lock before importing PIL/numpy

    def left() -> float:
        return float("inf") if deadline_wall is None else deadline_wall - time.time()

    if left() <= 0.5:
        return ""
    img = imaging.prepare(image_path)
    max_time = left() - _margin()
    if max_time <= 0.3:
        return ""
    raw = backend.read_text(img, build_prompt(), MAX_NEW_TOKENS, None if max_time == float("inf") else max_time)
    return postprocess.clean(raw)


def _recv_line(conn: socket.socket, limit_s: float = 10.0) -> bytes:
    end = time.monotonic() + limit_s
    buf = b""
    while not buf.endswith(b"\n"):
        rem = end - time.monotonic()
        if rem <= 0:
            break
        conn.settimeout(rem)
        part = conn.recv(65536)
        if not part:
            break
        buf += part
    return buf


def handle(conn: socket.socket, backend) -> bool:
    """1 件さばく。止める要求なら True。"""
    req = json.loads(_recv_line(conn).decode("utf-8"))
    op = req.get("op", "read")
    if op == "shutdown":
        conn.sendall(b'{"ok": true}\n')
        return True
    if op == "ping":
        conn.sendall((json.dumps({"ok": True, "pid": os.getpid()}) + "\n").encode())
        return False
    text = ""
    try:
        dl = req.get("deadline")
        text = read_one(backend, str(req.get("image_path", "")), float(dl) if dl is not None else None)
    except Exception:  # noqa: BLE001 - どの失敗も空の答え（常駐は生き続ける）
        traceback.print_exc()
        text = ""
    conn.settimeout(5)
    conn.sendall((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
    return False


def _serve(sdir: str) -> int:
    sdir = os.path.abspath(sdir or state_dir())
    os.makedirs(sdir, exist_ok=True)
    ready = os.path.join(sdir, "ready")
    for stale in (ready, os.path.join(sdir, "server_status.json")):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
    status_path = os.path.join(sdir, "server_status.json")
    log(f"loading backend={os.environ.get('X052_BACKEND', 'transformers')}")
    t0 = time.monotonic()
    try:
        backend = make_backend()
        backend.load()
        backend.warmup()
    except BaseException as e:  # noqa: BLE001 - 読み込みに失敗した常駐は、理由を残して終わる（supervise が起こし直す）
        traceback.print_exc()
        _write_json(status_path, {"state": "load_failed", "error": f"{type(e).__name__}: {e}",
                                  "runtime_env": runtime_env()})
        return 1
    load_s = round(time.monotonic() - t0, 2)
    log(f"backend loaded in {load_s}s")

    path = client.default_sock_path(sdir)
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    with open(os.path.join(sdir, "sock_path"), "w") as f:
        f.write(path)
    with open(os.path.join(sdir, "server.pid"), "w") as f:
        f.write(str(os.getpid()))
    try:
        env = runtime_env()
    except BaseException:  # noqa: BLE001 - 記録の失敗で常駐を止めない
        env = {}
    _write_json(status_path, {"state": "ready", "pid": os.getpid(), "load_seconds": load_s,
                              "warnings": list(getattr(backend, "warnings", [])), "runtime_env": env})
    with open(ready, "w") as f:  # 待ち受けを始めてから ready を書く
        f.write(str(time.time()))
    log(f"ready on {path}")
    try:
        while True:
            conn, _ = srv.accept()
            try:
                if handle(conn, backend):
                    break
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            finally:
                conn.close()
    finally:
        srv.close()
        for p in (path, ready):
            try:
                os.unlink(p)
            except OSError:
                pass
    return 0


def serve(sdir: Optional[str] = None) -> int:
    sdir = os.path.abspath(sdir or state_dir())
    with process_lock(sdir, "server") as lock:
        if lock is None:
            return 0
        return _serve(sdir)


if __name__ == "__main__":
    raise SystemExit(serve())
