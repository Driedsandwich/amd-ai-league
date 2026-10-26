#!/usr/bin/env python3
"""層を土台に積んで送り（crane append）、起動の命令などを付ける（crane mutate）。

送り先は環境変数 X052_IMAGE_REF から読む（画面にもファイルにも書き出さない）。
  --local : 手元の登録所（127.0.0.1 / localhost）へ試す形。--insecure・土台は alpine:3.20。
  手元でない送り先へは --i-have-human-approval が無ければ、送らずに止まる（公開は本人の承認が要る）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional

PROD_BASE = "docker.io/rocm/pytorch@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c"
LOCAL_BASE = "alpine:3.20"
REF_ENV = "X052_IMAGE_REF"


def ref_host(ref: str) -> str:
    first = ref.split("/", 1)[0]
    return first.rsplit(":", 1)[0] if ":" in first and not first.startswith("[") else first


def is_local_ref(ref: str) -> bool:
    return ref_host(ref) in ("127.0.0.1", "localhost", "[::1]")


DEFAULT_CMD = ["sleep", "infinity"]  # 試合 3 の既定。試合 2 の箱は --cmd python3 --cmd /app/supervise.py


def mutate_flags(model_name: str, cmd: Optional[List[str]] = None) -> List[str]:
    env = [("PYTHONPATH", "/opt/x052/site"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
           ("X052_BACKEND", "transformers"), ("X052_MODEL_DIR", f"/models/{model_name}")]
    flags = [x for c in (cmd or DEFAULT_CMD) for x in ("--cmd", c)] + ["-w", "/app"]
    for k, v in env:
        flags += ["-e", f"{k}={v}"]
    return flags


def run_crane(crane: str, args: List[str], secret: str) -> subprocess.CompletedProcess:
    r = subprocess.run([crane, *args], capture_output=True, text=True)
    r.stdout = r.stdout.replace(secret, "<X052_IMAGE_REF>")
    r.stderr = r.stderr.replace(secret, "<X052_IMAGE_REF>")
    return r


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", "-f", action="append", required=True, help="層の tar（順に積む）")
    ap.add_argument("--base", help="土台（既定: 本番は rocm/pytorch の digest・--local は alpine:3.20）")
    ap.add_argument("--model-name", required=True, help="/models/<名前> の <名前>")
    ap.add_argument("--cmd", action="append", default=None,
                    help="起動の命令の 1 語（繰り返し可。既定は sleep infinity）。例 --cmd python3 --cmd /app/supervise.py")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--i-have-human-approval", action="store_true")
    ap.add_argument("--crane", default=os.environ.get("X052_CRANE", "crane"))
    ap.add_argument("--platform", default="linux/amd64")
    a = ap.parse_args(argv)

    ref = os.environ.get(REF_ENV, "")
    if not ref:
        print(f"{REF_ENV} が未設定です。送りません。", file=sys.stderr)
        return 2
    local_dest = is_local_ref(ref)
    if a.local and not local_dest:
        print("--local は手元の登録所（127.0.0.1 / localhost）にだけ使えます。送りません。", file=sys.stderr)
        return 2
    if not local_dest and not a.i_have_human_approval:
        print("送り先は手元の登録所ではありません。--i-have-human-approval（本人の承認）が無いので、送らずに止まります。",
              file=sys.stderr)
        return 3
    base = a.base or (LOCAL_BASE if a.local else PROD_BASE)
    insecure = ["--insecure"] if a.local else []

    r = run_crane(a.crane, ["append", "--platform", a.platform, *insecure, "-b", base,
                            *[x for p in a.layer for x in ("-f", p)], "-t", ref], ref)
    if r.returncode != 0:
        print("crane append が失敗しました:", r.stderr.strip()[-600:], file=sys.stderr)
        return 1
    r = run_crane(a.crane, ["mutate", "--platform", a.platform, *insecure, *mutate_flags(a.model_name, a.cmd),
                            "-t", ref, ref], ref)
    if r.returncode != 0:
        print("crane mutate が失敗しました:", r.stderr.strip()[-600:], file=sys.stderr)
        return 1
    d = run_crane(a.crane, ["digest", *insecure, "--platform", a.platform, ref], ref)
    print("pushed; digest", d.stdout.strip() if d.returncode == 0 else "(unknown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
