#!/usr/bin/env python3
"""採点を真似て通しで流す（常駐の起動 → 画像 1 枚ごとに新しいプロセスで app.py）。

  python3 tools/run_images.py [--env X052_BACKEND=stub] [--report out.json]

測る物: 起動（supervise.py を裏で起こして ready が出るまで）の時間・各画像の時間・出力 JSON の形・
正規化した答えの一致・合計の時間・VRAM（mc3 の run10.py の VramSampler を使う。測れなければ n/a）。
門: 起動 <= 600 秒・各画像 <= 30 秒・合計 <= 600 秒・出力が全部有効・VRAM が 1〜48.48GiB（測れたとき）。
答えの一致は門に入れない（表に出す）。終了コードは門が全部合格（n/a は除く）なら 0。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOX = HERE.parent
DEFAULT_SAMPLES_DIR = BOX.parent.parent / "_src" / "mc2_samples"
RUN10 = BOX.parent / "mc3" / "tools" / "run10.py"


def normalize(s: str) -> str:
    """主催の正規化: 大文字・空白を全部除く・- . · _ を除く。"""
    return re.sub(r"[-.·_]", "", re.sub(r"\s+", "", (s or "").upper()))


def check_output_shape(path: Path):
    """(有効か, 理由, 中身)。text が文字列なら有効。confidence があるなら 0.0〜1.0 の数。"""
    if not path.exists():
        return False, "file missing", None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"invalid JSON: {e}", None
    if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
        return False, "text missing or not a string", obj
    if "confidence" in obj:
        c = obj["confidence"]
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
            return False, "confidence is not a number in 0..1", obj
    return True, "ok", obj


def load_vram_sampler():
    """mc3 の run10.py の VramSampler を、ファイルの場所から読む（sys.path を汚さない）。取れなければ None。"""
    try:
        spec = importlib.util.spec_from_file_location("mc3_run10", RUN10)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.VramSampler()
    except Exception:  # noqa: BLE001
        return None


def stop_group(proc: subprocess.Popen) -> None:
    """supervise を先に止め（子の server も止まる）、残りを process group ごと落とす。"""
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", type=Path, default=BOX / "app" / "app.py")
    ap.add_argument("--supervise", type=Path, default=BOX / "app" / "supervise.py")
    ap.add_argument("--samples-json", type=Path, default=HERE / "samples.json")
    ap.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    ap.add_argument("--state-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--startup-budget", type=float, default=600)
    ap.add_argument("--image-budget", type=float, default=30)
    ap.add_argument("--total-budget", type=float, default=600)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE（子の環境に足す）")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--keep-server", action="store_true")
    a = ap.parse_args(argv)

    samples = json.loads(a.samples_json.read_text(encoding="utf-8"))["samples"]
    work = Path(tempfile.mkdtemp(prefix="mc2run_"))  # 短い場所（unix socket のパスが長すぎないように）
    state_dir = a.state_dir or work / "state"
    out_dir = a.output_dir or work / "output"
    in_dir = work / "input"
    for d in (state_dir, out_dir, in_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    for s in samples:  # /app/input に画像を置く形を真似る
        shutil.copyfile(a.samples_dir / s["file"], in_dir / s["file"])
    env = dict(os.environ, X052_STATE_DIR=str(state_dir), X052_OUTPUT_DIR=str(out_dir))
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    sampler = load_vram_sampler()
    if sampler:
        sampler.start()
    logf = open(Path(state_dir) / "supervise.log", "ab")
    sup = subprocess.Popen([a.python, str(a.supervise)], env=env, stdin=subprocess.DEVNULL,
                           stdout=logf, stderr=logf, start_new_session=True)
    rows, startup = [], {}
    total = 0.0
    try:
        t0 = time.monotonic()
        ready_p = Path(state_dir) / "ready"
        state, detail = "timeout", ""
        while time.monotonic() - t0 < a.startup_budget:
            if ready_p.exists():
                state = "ready"
                break
            if sup.poll() is not None:
                state, detail = "supervise_exited", f"exit code {sup.returncode}"
                break
            time.sleep(0.2)
        startup = {"seconds": time.monotonic() - t0, "state": state, "detail": detail}
        for s in samples:
            name = s["file"]
            out = Path(out_dir) / f"{Path(name).stem}_output.json"
            if out.exists():
                out.unlink()  # 前回の出力を読まない（この道具自身の出力）
            t1, timed_out, rc = time.monotonic(), False, None
            try:
                r = subprocess.run([a.python, str(a.app), "--input-image", str(in_dir / name)], env=env,
                                   capture_output=True, text=True, timeout=a.image_budget)
                rc = r.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            dt = time.monotonic() - t1
            total += dt
            valid, why, obj = check_output_shape(out)
            got = obj.get("text") if valid else None
            rows.append({"n": s["n"], "file": name, "seconds": dt, "timed_out": timed_out, "returncode": rc,
                         "valid": valid, "valid_reason": why, "text": got, "expected": s["expected"],
                         "match": bool(valid and normalize(got) == normalize(s["expected"]))})
    finally:
        if sampler:
            sampler.stop_ev.set()
            sampler.join(timeout=6)
        if not a.keep_server:
            stop_group(sup)

    vram = sampler.summary() if sampler else {"available": False, "peak_gib": None, "min_gib": None, "n": 0}
    gates = {
        "startup_time": startup["state"] == "ready" and startup["seconds"] <= a.startup_budget,
        "each_image_time": all(r["seconds"] <= a.image_budget and not r["timed_out"] for r in rows),
        "total_time": total <= a.total_budget,
        "outputs_valid": all(r["valid"] for r in rows),
        "vram_1_to_48gib": (None if not vram["available"]
                            else (1.0 <= vram["peak_gib"] <= 48 * 1.01)),
    }
    correct = sum(1 for r in rows if r["match"])
    report = {"startup": startup, "images": rows, "total_image_seconds": total, "vram": vram,
              "gates": gates, "match": correct, "n": len(rows)}
    if a.report:
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"起動: {startup['seconds']:.1f}s（予算 {a.startup_budget:.0f}s）状態={startup['state']} {startup['detail']}".rstrip())
    print(f"{'N':>3} {'秒':>6} {'形':>3} {'一致':>4}  読んだ文字 / 期待")
    for r in rows:
        print(f"{r['n']:>3} {r['seconds']:>6.2f} {'OK' if r['valid'] else 'NG':>3} "
              f"{'OK' if r['match'] else 'NG':>4}  {r['text']!r} / {r['expected']!r}")
    print(f"全体（画像の合計）: {total:.1f}s（予算 {a.total_budget:.0f}s）  一致: {correct}/{len(rows)}")
    if vram["available"]:
        print(f"VRAM: 峰 {vram['peak_gib']:.2f}GiB / 最小 {vram['min_gib']:.2f}GiB（{vram['n']} 点）")
    else:
        print("VRAM: n/a（amd-smi / rocm-smi なし、または値を取れず）")
    for k, v in gates.items():
        print(f"門 {k}: {'n/a' if v is None else ('合格' if v else '不合格')}")
    return 0 if all(v is not False for v in gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
