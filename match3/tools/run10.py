#!/usr/bin/env python3
"""採点を真似て通しで流す（索引 1 回→問ごとに新しいプロセス）。

  python3 tools/run10.py --app app/app.py --corpus <束> --queries sample-questions.json \
      --index-dir <索引の場所> --output-dir <出力の場所> --python <python> [--report out.json]

測る物: 索引の時間・各問の時間・出力 JSON の形・答えの一致（answer_aliases も可・前後の空白／大文字小文字／
空白／° の違いは無視）・引用の集合の完全一致・問いの合間の常駐の生死・VRAM（amd-smi か rocm-smi があるとき）。
門: 索引 <= 索引の予算・各問 <= 30 秒・全体 <= 600 秒・出力が全部有効・VRAM が 1〜48GiB（測れなければ n/a）。
答えの正誤は門に入れない（表に出す）。終了コードは門が全部合格なら 0。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = 1024 ** 3
UNITS = {"b": 1, "kb": 1024, "kib": 1024, "mb": 1024 ** 2, "mib": 1024 ** 2,
         "gb": GIB, "gib": GIB, "tb": 1024 ** 4, "tib": 1024 ** 4}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").replace("°", "")).lower()


def answer_matches(got: str, expected: str, aliases) -> bool:
    cands = {norm(expected), *(norm(a) for a in (aliases or []))}
    return norm(got) in cands


def check_output_shape(path: Path):
    """(有効か, 理由, 中身)"""
    if not path.exists():
        return False, "file missing", None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"invalid JSON: {e}", None
    if not isinstance(obj, dict) or not isinstance(obj.get("answer"), str):
        return False, "answer missing or not a string", obj
    c = obj.get("citations")
    if not isinstance(c, list) or not all(isinstance(x, str) for x in c):
        return False, "citations missing or not a list of strings", obj
    return True, "ok", obj


# ---- VRAM ----
def _to_bytes(val, unit=None):
    try:
        if isinstance(val, dict):
            return _to_bytes(val.get("value"), val.get("unit"))
        m = re.match(r"^\s*([0-9.]+)\s*([A-Za-z]*)\s*$", str(val))
        if not m:
            return None
        num, u = float(m.group(1)), (unit or m.group(2) or "b").lower()
        return num * UNITS.get(u, 1)
    except Exception:  # noqa: BLE001
        return None


def parse_vram_used(text: str):
    """amd-smi / rocm-smi の JSON から使用中の VRAM（byte・全 GPU の合計）を取り出す。取れなければ None。"""
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return None
    found = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                if "vram" in kl and "used" in kl and "total" not in kl.replace("total used", "used"):
                    b = _to_bytes(v)
                    if b is not None:
                        found.append(b)
                        continue
                if kl in ("used_vram", "vram_used") or "total used memory" in kl:
                    b = _to_bytes(v)
                    if b is not None:
                        found.append(b)
                        continue
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return sum(found) if found else None


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_ev = threading.Event()
        self.samples = []
        if shutil.which("amd-smi"):
            self.cmd = ["amd-smi", "metric", "--mem-usage", "--json"]
        elif shutil.which("rocm-smi"):
            self.cmd = ["rocm-smi", "--showmeminfo", "vram", "--json"]
        else:
            self.cmd = None

    def run(self):
        while self.cmd and not self.stop_ev.is_set():
            try:
                r = subprocess.run(self.cmd, capture_output=True, text=True, timeout=5)
                v = parse_vram_used(r.stdout)
                if v is not None:
                    self.samples.append(v)
            except Exception:  # noqa: BLE001
                pass
            self.stop_ev.wait(1.0)

    def summary(self):
        if not self.cmd or not self.samples:
            return {"available": False, "peak_gib": None, "min_gib": None, "n": 0}
        return {"available": True, "peak_gib": max(self.samples) / GIB,
                "min_gib": min(self.samples) / GIB, "n": len(self.samples)}


def server_alive(index_dir: Path):
    pidf = index_dir / "server.pid"
    try:
        pid = int(pidf.read_text().strip())
    except Exception:  # noqa: BLE001
        return None  # 常駐の記録なし
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--index-budget", type=float, default=600)
    ap.add_argument("--query-budget", type=float, default=30)
    ap.add_argument("--total-budget", type=float, default=600)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE（子の環境に足す）")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--keep-server", action="store_true")
    a = ap.parse_args(argv)

    qs = json.loads(a.queries.read_text(encoding="utf-8"))["queries"]
    a.index_dir.mkdir(parents=True, exist_ok=True)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(X052_INDEX_DIR=str(a.index_dir), MC3_INDEX_DIR=str(a.index_dir),
               X052_OUTPUT_DIR=str(a.output_dir), MC2_OUTPUT_DIR=str(a.output_dir))
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    sampler = VramSampler()
    sampler.start()
    rows, index_info = [], {}
    try:
        t0 = time.time()
        try:
            r = subprocess.run([a.python, str(a.app), "--index", str(a.corpus)], env=env,
                               capture_output=True, text=True, timeout=a.index_budget)
            index_info = {"seconds": time.time() - t0, "returncode": r.returncode,
                          "timed_out": False, "stderr_tail": r.stderr[-300:]}
        except subprocess.TimeoutExpired:
            index_info = {"seconds": time.time() - t0, "returncode": None, "timed_out": True, "stderr_tail": ""}
        total = 0.0
        for q in qs:
            qid = f"query_{q['n']:02d}"
            out = a.output_dir / f"{qid}_output.json"
            if out.exists():
                out.unlink()  # 前回の出力を読まない（この道具自身の出力）
            alive_before = server_alive(a.index_dir)
            t1, timed_out, rc = time.time(), False, None
            try:
                r = subprocess.run([a.python, str(a.app), "--corpus", str(a.corpus), "--query-id", qid,
                                    "--query", q["query"]], env=env, capture_output=True, text=True,
                                   timeout=a.query_budget)
                rc = r.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            dt = time.time() - t1
            total += dt
            valid, why, obj = check_output_shape(out)
            exp_cit = set(q.get("expected_citations", []))
            ans_ok = bool(valid and answer_matches(obj["answer"], q.get("expected_answer", ""),
                                                   q.get("answer_aliases")))
            cit_ok = bool(valid and set(obj["citations"]) == exp_cit)
            rows.append({"n": q["n"], "seconds": dt, "timed_out": timed_out, "returncode": rc,
                         "valid": valid, "valid_reason": why, "answer": obj.get("answer") if obj else None,
                         "citations": obj.get("citations") if obj else None,
                         "answer_ok": ans_ok, "citations_ok": cit_ok, "correct": ans_ok and cit_ok,
                         "server_alive_before": alive_before})
        final_alive = server_alive(a.index_dir)
    finally:
        sampler.stop_ev.set()
        sampler.join(timeout=6)
        if not a.keep_server:
            try:
                os.kill(int((a.index_dir / "server.pid").read_text().strip()), signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass

    vram = sampler.summary()
    gates = {
        "index_time": index_info["seconds"] <= a.index_budget and not index_info["timed_out"],
        "each_query_time": all(r["seconds"] <= a.query_budget and not r["timed_out"] for r in rows),
        "total_time": total <= a.total_budget,
        "outputs_valid": all(r["valid"] for r in rows),
        "vram_1_to_48gib": (None if not vram["available"]
                            else (vram["peak_gib"] >= 1.0 and vram["peak_gib"] <= 48 * 1.01)),
    }
    score = sum(1 for r in rows if r["correct"])
    report = {"index": index_info, "queries": rows, "total_query_seconds": total, "vram": vram,
              "gates": gates, "correct": score, "n": len(rows), "server_alive_at_end": final_alive}
    if a.report:
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"索引: {index_info['seconds']:.1f}s (予算 {a.index_budget:.0f}s) rc={index_info['returncode']}")
    print(f"{'Q':>3} {'秒':>6} {'形':>3} {'答え':>4} {'引用':>4} {'常駐':>4}  答え / 引用")
    for r in rows:
        alive = {True: "生", False: "死", None: "-"}[r["server_alive_before"]]
        print(f"{r['n']:>3} {r['seconds']:>6.2f} {'OK' if r['valid'] else 'NG':>3} "
              f"{'OK' if r['answer_ok'] else 'NG':>4} {'OK' if r['citations_ok'] else 'NG':>4} {alive:>4}  "
              f"{r['answer']!r} / {r['citations']}")
    print(f"全体（問いの合計）: {total:.1f}s (予算 {a.total_budget:.0f}s)   正解（答え＋引用）: {score}/{len(rows)}")
    if vram["available"]:
        print(f"VRAM: 峰 {vram['peak_gib']:.2f}GiB / 最小 {vram['min_gib']:.2f}GiB（{vram['n']} 点）")
    else:
        print("VRAM: n/a（amd-smi / rocm-smi なし、または値を取れず）")
    for k, v in gates.items():
        print(f"門 {k}: {'n/a' if v is None else ('合格' if v else '不合格')}")
    return 0 if all(v is not False for v in gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
