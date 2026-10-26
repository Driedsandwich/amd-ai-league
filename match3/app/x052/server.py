"""常駐。unix socket で待ち、模型と索引を持ったまま問いに答える。

起動の順: 模型を読む→画像を 1 枚ずつ順に（別の糸を使わず）文字にする→索引を作る→ready を書く→待つ。
画像は生成に max_time をかけて 1 枚の上限を守る。1 枚が時間切れでも残りは続け、全体の期限を過ぎたときだけ飛ばす。
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tempfile
import time
import traceback
from typing import List, Optional

from . import indexer, parsers
from .answerer import Answerer
from .backend import make_backend
from .retrieve import ID_RE

IMAGE_TOTAL_BUDGET_S = float(os.environ.get("X052_IMAGE_TOTAL_S", "240"))
IMAGE_EACH_BUDGET_S = float(os.environ.get("X052_IMAGE_EACH_S", "20"))
STARTUP_RESERVE_S = 25.0  # 画像の後（索引の組み立て・ready）のために残す秒


def sock_path_for(index_dir: str) -> str:
    """unix socket のパスは約 104 字まで。長いときは短い場所に置き、sock_path に記録する。"""
    p = os.path.join(index_dir, "x052.sock")
    if len(p.encode()) < 100:
        return p
    h = hashlib.sha1(os.path.abspath(index_dir).encode()).hexdigest()[:10]
    q = os.path.join(tempfile.gettempdir(), f"x052-{h}.sock")
    if len(q.encode()) >= 100:
        q = f"/tmp/x052-{h}.sock"
    return q


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _runtime_env() -> dict:
    """本番の像で動いた torch・numpy・PIL の版と置き場所と GPU を残す（依存の層が土台の ROCm 版を影にしていないかを後から読むため）。"""
    env = {"python": sys.version.split()[0], "pythonpath": os.environ.get("PYTHONPATH", "")}
    for name in ("torch", "numpy", "PIL"):
        try:
            mod = __import__(name)
            env[name] = {"version": getattr(mod, "__version__", "?"), "file": getattr(mod, "__file__", "?")}
        except BaseException as e:  # noqa: BLE001 - 試験の模型では torch が無いことがある
            env[name] = {"error": f"{type(e).__name__}: {e}"}
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["hip"] = getattr(torch.version, "hip", None)
        env["device"] = torch.cuda.get_device_name(0) if env["cuda_available"] else None
    except BaseException as e:  # noqa: BLE001
        env["cuda_error"] = f"{type(e).__name__}: {e}"
    return env


def add_image_chunks(index_dir: str, corpus: str, backend, chunks: List[dict],
                     each: Optional[float] = None, total: Optional[float] = None) -> List[dict]:
    """画像を 1 枚ずつ順に文字にして塊に足す。結果の一覧（画像ごとの状態）を返す。

    each＝1 枚の上限秒（生成の max_time）・total＝全体の上限秒。1 枚の時間切れでは残りを飛ばさない。
    """
    each = IMAGE_EACH_BUDGET_S if each is None else each
    total = IMAGE_TOTAL_BUDGET_S if total is None else total
    results: List[dict] = []
    if not getattr(backend, "has_vision", False):
        return results
    try:
        manifest = json.load(open(os.path.join(index_dir, "manifest.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return results
    images = [m["path"] for m in manifest if m.get("status") == "image"]
    t0 = time.monotonic()
    for rel in images:
        remain = total - (time.monotonic() - t0)
        if remain <= 1:
            results.append({"path": rel, "status": "skipped_time"})
            continue
        t1 = time.monotonic()
        limit = min(each, remain)
        try:
            text = backend.describe_image(os.path.join(corpus, rel), max_time=limit)
            dt = time.monotonic() - t1
            status = "timeout" if dt >= limit - 0.05 else "ok"
        except Exception as e:  # noqa: BLE001
            log(f"image failed: {rel}: {type(e).__name__}: {e}")
            results.append({"path": rel, "status": "failed", "error": f"{type(e).__name__}: {e}"})
            continue
        if text and text.strip():
            for k, piece in enumerate(parsers.chunk_plain(text)):
                chunks.append({"id": len(chunks), "path": rel, "loc": f"image text {k + 1}", "text": piece})
            if status == "timeout":
                status = "timeout_partial"
        results.append({"path": rel, "status": status, "seconds": round(dt, 2)})
    return results


def load_ids(index_dir: str, chunks: List[dict], n_base: int):
    """索引の時に保存した識別子の表（ids.json）を読み、画像から起こした塊の分を足す。読めなければ None（作り直す）。"""
    try:
        table = json.load(open(os.path.join(index_dir, "ids.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for i in range(n_base, len(chunks)):
        for ident in dict.fromkeys(ID_RE.findall(chunks[i]["text"])):
            table.setdefault(ident, []).append(i)
    return table


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


def handle(conn: socket.socket, answerer: Answerer) -> bool:
    """1 件さばく。常駐を止める要求なら True。"""
    req = json.loads(_recv_line(conn).decode("utf-8"))
    op = req.get("op", "query")
    if op == "shutdown":
        conn.sendall(b'{"ok": true}\n')
        return True
    if op == "ping":
        conn.sendall((json.dumps({"ok": True, "pid": os.getpid(), "chunks": len(answerer.chunks)}) + "\n").encode())
        return False
    empty = {"answer": "", "citations": [], "confidence": 0.0}
    deadline = None
    if req.get("expires_wall") is not None:
        # 壁時計の期限を、この常駐の単調な時計の期限に直す。過ぎた問い（待たされた間に問い側が諦めた）は捨てる。
        deadline = time.monotonic() + (float(req["expires_wall"]) - time.time())
        if deadline - time.monotonic() <= 0.3:
            conn.sendall((json.dumps(empty) + "\n").encode())
            return False
    try:
        ans, cites, conf = answerer.answer(str(req.get("query", "")), deadline=deadline)
        resp = {"answer": ans, "citations": cites, "confidence": conf}
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        resp = empty
    conn.settimeout(5)
    conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
    return False


def serve(index_dir: str, corpus: Optional[str] = None) -> int:
    index_dir = os.path.abspath(index_dir)
    corpus = corpus or os.environ.get("X052_CORPUS", "/app/corpus")
    ready = os.path.join(index_dir, "ready")
    try:
        os.unlink(ready)
    except FileNotFoundError:
        pass
    status_path = os.path.join(index_dir, "server_status.json")
    log(f"loading backend={os.environ.get('X052_BACKEND', 'transformers')}")
    t0 = time.monotonic()
    try:
        backend = make_backend()
        backend.load()
    except BaseException as e:  # noqa: BLE001 - 読み込みに失敗した常駐は、理由を残して終わる（問いは空の答えになる）
        traceback.print_exc()
        _write_json(status_path, {"state": "load_failed", "error": f"{type(e).__name__}: {e}"})
        return 1
    warnings = list(getattr(backend, "warnings", []))
    if warnings:
        _write_json(os.path.join(index_dir, "warnings.json"), warnings)
    log(f"backend loaded in {time.monotonic() - t0:.1f}s")
    try:
        _write_json(os.path.join(index_dir, "runtime_env.json"), _runtime_env())
    except BaseException:  # noqa: BLE001 - 記録の失敗で常駐を止めない
        traceback.print_exc()

    chunks = indexer.load_chunks(index_dir)
    n_base = len(chunks)
    total = IMAGE_TOTAL_BUDGET_S
    if os.environ.get("X052_STARTUP_DEADLINE_WALL"):
        left = float(os.environ["X052_STARTUP_DEADLINE_WALL"]) - time.time() - STARTUP_RESERVE_S
        total = max(0.0, min(total, left))
    img = add_image_chunks(index_dir, corpus, backend, chunks, total=total)
    _write_json(os.path.join(index_dir, "images_status.json"), img)
    log(f"chunks={len(chunks)} images={len(img)} "
        f"ok={sum(1 for r in img if r['status'] == 'ok')} (total budget {total:.0f}s)")

    answerer = Answerer(chunks, backend, load_ids(index_dir, chunks, n_base))
    path = sock_path_for(index_dir)
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    with open(os.path.join(index_dir, "sock_path"), "w") as f:
        f.write(path)
    with open(os.path.join(index_dir, "server.pid"), "w") as f:
        f.write(str(os.getpid()))
    _write_json(status_path, {"state": "ready", "chunks": len(chunks), "images": img})
    with open(ready, "w") as f:  # 画像まで全部済んでから ready を書く
        f.write(str(time.time()))
    log(f"ready on {path}")
    try:
        while True:
            conn, _ = srv.accept()
            try:
                if handle(conn, answerer):
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
