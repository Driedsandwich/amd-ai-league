#!/usr/bin/env python3
"""AMD Mini-Challenge 2（文字の読み取り）提出の入口。主催の呼び方に従う。

  python3 /app/app.py --input-image /app/input/image_01.png

出力は /app/output/<入力の名前の拡張子を除いた物>_output.json に {"text": "..."}。
どんな失敗（常駐が無い・死んでいる・時間切れ・例外）でも {"text": ""} を書いて、終了コード 0 で終わる。
常駐（supervise.py が起こす server.py）が模型を持っているので、この入口は薄い。

環境変数: X052_IMAGE_BUDGET（始まりから数える絶対の締切・秒・既定 27）・X052_READY_MARGIN（常駐の ready を待つのは
締切のこの秒前まで・既定 2）・X052_ALARM_S（最後の砦・既定 締切＋1）・X052_STATE_DIR（既定 /app/state）・
X052_OUTPUT_DIR（既定 /app/output）。X052_TEST_SUPERVISE_PYTHON は試験用の起動 Python 差し替え。
"""
from __future__ import annotations

import time

T0 = time.monotonic()  # この処理の始まり
WALL0 = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def output_dir() -> Path:
    return Path(os.environ.get("X052_OUTPUT_DIR") or "/app/output")


def state_dir() -> str:
    return os.environ.get("X052_STATE_DIR") or "/app/state"


def output_name(image_path: str) -> str:
    """image_01.png -> image_01_output.json（拡張子は大文字小文字を問わず除く）。"""
    return f"{Path(image_path).stem}_output.json"


def write_output(image_path: str, text: str) -> None:
    """一時のファイルに書いてから置き換える（途中まで書いた物を読ませない）。"""
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = output_name(image_path)
    tmp = out_dir / f".{name}.tmp"
    tmp.write_text(json.dumps({"text": text if isinstance(text, str) else ""}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_dir / name)


def write_output_safely(image_path: str, text: str) -> bool:
    try:
        write_output(image_path, text)
        return True
    except Exception:  # noqa: BLE001
        pass
    try:  # 別の書き方: 直に書く
        out_dir = output_dir()
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir / output_name(image_path), "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": text if isinstance(text, str) else ""}, ensure_ascii=False))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"could not write output: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def _start_supervisor(sdir: str, wait_until: float) -> None:
    from mc2ocr import client
    from mc2ocr.lifecycle import process_lock

    # No waiting for another CLI, a loading server, or a supervisor in backoff.
    with process_lock(sdir, "supervise") as supervisor_lock:
        if supervisor_lock is None:
            return
        with process_lock(sdir, "server") as server_lock:
            if server_lock is None or client.is_ready(sdir) or time.monotonic() >= wait_until:
                return
        # Release the server probe before the child can launch its server.
        with open(Path(sdir) / "supervise.log", "ab", buffering=0) as log:
            subprocess.Popen(
                [os.environ.get("X052_TEST_SUPERVISE_PYTHON") or "python3",
                 str(Path(__file__).resolve().with_name("supervise.py")),
                 "--lock-fd", str(supervisor_lock.fileno())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(supervisor_lock.fileno(),),
            )


def run(image_path: str) -> None:
    budget = _envf("X052_IMAGE_BUDGET", 27)
    alarm_s = _envf("X052_ALARM_S", budget + 1)
    ready_margin = _envf("X052_READY_MARGIN", 2)
    deadline_wall = WALL0 + budget

    def last_resort(signum, frame):  # 最後の砦: 空の答えを書いて終わる
        write_output_safely(image_path, "")
        os._exit(0)

    try:
        signal.signal(signal.SIGALRM, last_resort)
        signal.setitimer(signal.ITIMER_REAL, max(0.05, alarm_s - (time.monotonic() - T0)))
    except Exception:  # noqa: BLE001
        pass
    text = ""
    try:
        from mc2ocr import client

        sdir = state_dir()
        wait_until = T0 + budget - ready_margin
        if not client.is_ready(sdir) and time.monotonic() < wait_until:
            # Even a stalled spawn must leave time to write the usual empty output.
            startup_end = min(wait_until, T0 + budget, T0 + alarm_s)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, startup_end - time.monotonic()))
            try:
                _start_supervisor(sdir, wait_until)
            except Exception:  # noqa: BLE001 - failed startup still uses the normal wait/deadline
                pass
            finally:
                signal.setitimer(signal.ITIMER_REAL, max(0.001, T0 + alarm_s - time.monotonic()))
        while not client.is_ready(sdir) and time.monotonic() < wait_until:
            time.sleep(0.2)  # 常駐がまだ模型を読んでいる間は、締切の少し前まで待つ
        resp = client.read_image(sdir, str(Path(image_path).resolve()), deadline_wall)
        if resp and isinstance(resp.get("text"), str):
            text = resp["text"]
    except Exception:  # noqa: BLE001
        text = ""
    write_output_safely(image_path, text)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image")
    try:
        args, _ = ap.parse_known_args()
    except SystemExit:
        return 0
    if not args.input_image:
        print("--input-image is required", file=sys.stderr)
        return 0
    try:
        run(args.input_image)
    except Exception as e:  # noqa: BLE001
        print(f"failed: {type(e).__name__}: {e}", file=sys.stderr)
        write_output_safely(args.input_image, "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
