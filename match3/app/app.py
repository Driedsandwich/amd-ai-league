#!/usr/bin/env python3
"""AMD Mini-Challenge 3（RAG）提出の入口。主催の呼び方 2 つに従う。

  python3 /app/app.py --index /app/corpus
  python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "…"

出力は /app/output/<query-id>_output.json。どんな失敗でも JSON を書いて 0 で終わる。

時間の決まり（環境変数で変えられる）:
  --index  : 始まりから X052_INDEX_DEADLINE 秒（480）の絶対の期限。文の取り出しには X052_PARSE_BUDGET 秒（240）。
             常駐を起こした後は、その終わりを見張り、ready と ping の疎通が取れたときだけ「準備よし」と
             INDEX_DIR/startup_status.json に書く。終了コードは常に 0。
  --query  : 始まりから X052_QUERY_BUDGET 秒（30）。残りから安全の余白 X052_QUERY_MARGIN 秒（4）を引いた分を
             常駐へ渡す。最後の砦として X052_ALARM_S 秒（28）で空の答えの JSON を書いて終わる。
"""
from __future__ import annotations

import time

T0 = time.monotonic()  # この処理の始まり（問いの 30 秒はここから数える）

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


def _env(*names, default=""):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def output_dir() -> Path:
    return Path(_env("X052_OUTPUT_DIR", "MC2_OUTPUT_DIR", default="/app/output"))


def index_dir() -> Path:
    return Path(_env("X052_INDEX_DIR", "MC3_INDEX_DIR", default="/app/index"))


def _payload(answer, citations, confidence) -> str:
    return json.dumps({
        "answer": answer if isinstance(answer, str) else "",
        "citations": [str(c) for c in (citations or [])],
        "confidence": float(confidence),
    }, ensure_ascii=False)


def write_output(query_id: str, answer: str, citations, confidence: float) -> None:
    """一時のファイルに書いてから置き換える（途中まで書いた物を読ませない）。"""
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{query_id}_output.json"
    tmp = out_dir / f".{query_id}_output.json.tmp"
    tmp.write_text(_payload(answer, citations, confidence), encoding="utf-8")
    os.replace(tmp, final)


def write_output_safely(query_id: str, answer: str, citations, confidence: float) -> bool:
    """書き込みの失敗も例外で捕まえ、別の書き方でもう一度書く。書けたら True。"""
    try:
        write_output(query_id, answer, citations, confidence)
        return True
    except BaseException as e:  # noqa: BLE001 - 時計の割り込み以外は何でも拾う
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
    try:  # 別の書き方: 直に書く（一時のファイルと置き換えを使わない）
        out_dir = output_dir()
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / f"{query_id}_output.json", "w", encoding="utf-8") as f:
            f.write(_payload(answer, citations, confidence))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"could not write output: {type(e).__name__}: {e}", file=sys.stderr)
        return False


# ---------------- --index ----------------
def _status(idx: Path, **kw) -> None:
    try:
        tmp = idx / "startup_status.json.tmp"
        tmp.write_text(json.dumps(kw, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, idx / "startup_status.json")
    except Exception:  # noqa: BLE001
        pass


def do_index(corpus: Path) -> None:
    from x052 import client, indexer

    start = time.monotonic()
    deadline_s = _envf("X052_INDEX_DEADLINE", 480)
    parse_budget = min(_envf("X052_PARSE_BUDGET", 240), max(1.0, deadline_s - 60))
    idx = index_dir()
    idx.mkdir(parents=True, exist_ok=True)
    client.shutdown(str(idx), timeout=2)  # 前の常駐が残っていれば止める
    for stale in ("ready", "sock_path", "server_status.json", "startup_status.json"):
        try:
            os.unlink(idx / stale)
        except FileNotFoundError:
            pass
    manifest = indexer.build_index(str(corpus), str(idx), parse_budget=parse_budget)
    info = {
        "parse_seconds": round(time.monotonic() - start, 2),
        "files_indexed": sum(1 for m in manifest if m.get("status") == "indexed"),
        "files_skipped_time": [m["path"] for m in manifest if m.get("reason") == "skipped_time"],
        "files_truncated": [m["path"] for m in manifest if m.get("truncated")],
    }
    if os.environ.get("X052_NO_SERVER") == "1":
        _status(idx, state="no_server", elapsed=round(time.monotonic() - start, 2), **info)
        return
    env = dict(os.environ, X052_CORPUS=str(corpus), X052_INDEX_DIR=str(idx))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["X052_STARTUP_DEADLINE_WALL"] = str(time.time() + (deadline_s - (time.monotonic() - start)))
    logf = open(idx / "server.log", "ab")
    proc = subprocess.Popen([sys.executable, str(APP_DIR / "app.py"), "--serve"], env=env,
                            stdin=subprocess.DEVNULL, stdout=logf, stderr=logf, start_new_session=True)
    state, detail = "timeout", ""
    while time.monotonic() - start < deadline_s:
        rc = proc.poll()
        if rc is not None:  # 常駐が終わった（読み込み失敗など）
            state, detail = "server_exited", f"exit code {rc}"
            break
        if (idx / "ready").exists():
            if client.ping(str(idx), timeout=3):
                state = "ready"
                break
        time.sleep(0.3)
    warnings = []
    try:
        warnings = json.loads((idx / "warnings.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    _status(idx, state=state, detail=detail, elapsed=round(time.monotonic() - start, 2),
            deadline_seconds=deadline_s, server_pid=proc.pid, warnings=warnings, **info)
    print(f"startup: {state} {detail}".strip(), file=sys.stderr)


# ---------------- --query ----------------
def do_query(query_id: str, query: str) -> None:
    budget = _envf("X052_QUERY_BUDGET", 30)
    margin = _envf("X052_QUERY_MARGIN", 4)
    alarm_s = _envf("X052_ALARM_S", 28)

    def last_resort(signum, frame):  # 最後の砦: 空の答えと空の引用を書いて終わる
        write_output_safely(query_id, "", [], 0.0)
        os._exit(0)

    try:
        signal.signal(signal.SIGALRM, last_resort)
        signal.setitimer(signal.ITIMER_REAL, max(0.05, alarm_s - (time.monotonic() - T0)))
    except Exception:  # noqa: BLE001
        pass
    answer, citations, confidence = "", [], 0.0
    try:
        from x052 import client
        remaining = budget - margin - (time.monotonic() - T0)
        resp = client.ask(str(index_dir()), query, timeout=max(0.0, remaining))
        if resp and isinstance(resp.get("answer"), str) and isinstance(resp.get("citations"), list):
            answer = resp["answer"]
            citations = [str(c) for c in resp["citations"]]
            confidence = float(resp.get("confidence") or 0.0)
            if not answer:
                citations = []
    except Exception:  # noqa: BLE001
        answer, citations, confidence = "", [], 0.0
    write_output_safely(query_id, answer, citations, confidence)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--serve", action="store_true", help="internal: run the resident server")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()

    if args.serve:
        from x052 import server
        return server.serve(str(index_dir()))
    if args.index is not None:
        try:
            do_index(args.index)
        except Exception as e:  # noqa: BLE001 - 索引が失敗しても 0 で終わる（問いは空の答えになる）
            print(f"index failed: {type(e).__name__}: {e}", file=sys.stderr)
            _status(index_dir(), state="index_failed", detail=f"{type(e).__name__}: {e}")
        return 0
    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")
    do_query(args.query_id, args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
