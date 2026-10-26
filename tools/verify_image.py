#!/usr/bin/env python3
"""送った像を照らす。送り先は環境変数 X052_IMAGE_REF（書き出さない）。全部合格なら終了コード 0。

  目録の下の層が土台と同じか（圧縮の識別子と config の diff_ids の両方）・linux/amd64・起動の命令・
  ENTRYPOINT 無し・作業場所・実行の利用者・環境変数・合計の大きさ・認証なしで目録／config／一番小さい自分の層が取れるか。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from push_image import DEFAULT_CMD, LOCAL_BASE, PROD_BASE, REF_ENV, is_local_ref  # noqa: E402

GIB = 1024 ** 3
WANT_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "X052_BACKEND": "transformers",
            "PYTHONPATH": "/opt/x052/site"}


class Crane:
    def __init__(self, crane: str, insecure: bool, secret: str, platform: str):
        self.crane, self.insecure, self.secret, self.platform = crane, insecure, secret, platform
        self._tmp = tempfile.TemporaryDirectory(prefix="x052_empty_docker_config_")  # 認証情報の無い空の設定
        self.cfg = self._tmp.name

    def _env(self):
        e = dict(os.environ)
        e["DOCKER_CONFIG"] = self.cfg
        return e

    def run(self, sub: str, ref: str, extra=()):
        args = [self.crane, sub, "--platform", self.platform, *(["--insecure"] if self.insecure else []), *extra, ref]
        r = subprocess.run(args, capture_output=True, env=self._env())
        return r.returncode, r.stdout, r.stderr.decode("utf-8", "replace").replace(self.secret, "<REF>")

    def json(self, sub: str, ref: str):
        rc, out, err = self.run(sub, ref)
        if rc != 0:
            raise RuntimeError(f"crane {sub} failed: {err.strip()[-300:]}")
        return json.loads(out)

    def blob_info(self, ref_with_digest: str, expect: str, limit: int) -> Tuple[int, str]:
        args = [self.crane, "blob", *(["--insecure"] if self.insecure else []), ref_with_digest]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=self._env())
        h, n = hashlib.sha256(), 0
        for chunk in iter(lambda: p.stdout.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if n > limit:
                p.kill()
                break
        p.wait()
        return n, "sha256:" + h.hexdigest()


def repo_of(ref: str) -> str:
    if "@" in ref:
        return ref.split("@", 1)[0]
    head, _, tail = ref.rpartition(":")
    return head if head and "/" not in tail else ref


def verify(ref: str, base: str, crane: str = "crane", insecure: bool = False, platform: str = "linux/amd64",
           layer_tars: Optional[List[str]] = None, base_gib: float = 29.2, limit_gib: float = 60.0,
           model_name: Optional[str] = None, cmd: Optional[List[str]] = None) -> List[Tuple[str, bool, str]]:
    res: List[Tuple[str, bool, str]] = []
    c = Crane(crane, insecure, ref, platform)
    base_c = Crane(crane, False, ref, platform)  # 土台は公開の登録所
    try:
        man = c.json("manifest", ref)
        cfg = c.json("config", ref)
        res.append(("認証なしで目録と config が取れる", True, f"layers={len(man['layers'])}"))
    except Exception as e:  # noqa: BLE001
        return [("認証なしで目録と config が取れる", False, str(e))]
    try:
        bman = base_c.json("manifest", base)
        bcfg = base_c.json("config", base)
    except Exception as e:  # noqa: BLE001
        return res + [("土台の目録を取る", False, str(e))]

    n = len(bman["layers"])
    own = man["layers"][n:]
    same_digest = [l["digest"] for l in man["layers"][:n]] == [l["digest"] for l in bman["layers"]]
    res.append(("目録の下の層が土台と同じ（圧縮の識別子）", same_digest, f"base layers={n}"))
    diff, bdiff = cfg["rootfs"]["diff_ids"], bcfg["rootfs"]["diff_ids"]
    res.append(("config の diff_ids の下が土台と同じ", diff[:len(bdiff)] == bdiff,
                f"diff_ids={len(diff)} base={len(bdiff)}"))
    res.append(("自分の層が積まれている（目録と diff_ids の数が合う）",
                len(own) >= 1 and len(diff) == len(man["layers"]), f"own layers={len(own)}"))
    res.append(("linux/amd64", cfg.get("os") == "linux" and cfg.get("architecture") == "amd64",
                f"{cfg.get('os')}/{cfg.get('architecture')}"))
    cf, bcf = cfg.get("config", {}), bcfg.get("config", {})
    want_cmd = list(cmd or DEFAULT_CMD)
    res.append(("起動の命令 sleep infinity" if want_cmd == DEFAULT_CMD else "起動の命令（--cmd で渡した物）",
                cf.get("Cmd") == want_cmd, f"Cmd={cf.get('Cmd')}（期待 {want_cmd}）"))
    res.append(("ENTRYPOINT が無い", not cf.get("Entrypoint"), f"Entrypoint={cf.get('Entrypoint')}"))
    res.append(("作業場所 /app", cf.get("WorkingDir") == "/app", f"WorkingDir={cf.get('WorkingDir')}"))
    user = cf.get("User", "")
    res.append(("実行の利用者が root か土台と同じ", user in ("", "root", "0", "0:0") or user == bcf.get("User"),
                f"User={user!r}（土台 {bcf.get('User')!r}）"))
    env = dict(e.split("=", 1) for e in cf.get("Env", []) if "=" in e)
    benv = dict(e.split("=", 1) for e in bcf.get("Env", []) if "=" in e)
    lost = [k for k in ("PATH", "VIRTUAL_ENV") if k in benv and env.get(k) != benv[k]]
    res.append(("土台の PATH と VIRTUAL_ENV が残っている", "PATH" in env and not lost,
                f"PATH={'あり' if 'PATH' in env else 'なし'} 失われた={lost}"))
    want = dict(WANT_ENV)
    if model_name:
        want["X052_MODEL_DIR"] = f"/models/{model_name}"
    elif "X052_MODEL_DIR" in env:
        want["X052_MODEL_DIR"] = env["X052_MODEL_DIR"]
    miss = {k: env.get(k) for k, v in want.items() if env.get(k) != v}
    if "X052_MODEL_DIR" not in want:
        miss["X052_MODEL_DIR"] = env.get("X052_MODEL_DIR")
    res.append(("自分の環境変数が足されている", not miss, f"足りない/違う={miss}"))
    if layer_tars:
        total = sum(os.path.getsize(p) for p in layer_tars) + base_gib * GIB
        res.append((f"合計が {limit_gib:g}GiB に収まる", total <= limit_gib * GIB,
                    f"{total / GIB:.3f}GiB（土台 {base_gib}GiB + 自分の層の tar）"))
    else:
        res.append((f"合計が {limit_gib:g}GiB に収まる", False, "--layer-tar が渡されていないので判定できない"))
    if own:
        small = min(own, key=lambda l: l["size"])
        got, dig = c.blob_info(f"{repo_of(ref)}@{small['digest']}", small["digest"], small["size"] + (1 << 20))
        res.append(("認証なしで一番小さい自分の層が取れる", got == small["size"] and dig == small["digest"],
                    f"{got} byte（目録 {small['size']}）"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="照らす土台（既定: 本番は rocm/pytorch の digest・--local は alpine:3.20）")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--layer-tar", action="append", default=[])
    ap.add_argument("--base-gib", type=float, default=29.2)
    ap.add_argument("--limit-gib", type=float, default=60.0)
    ap.add_argument("--model-name")
    ap.add_argument("--cmd", action="append", default=None,
                    help="push_image.py に渡した起動の命令（繰り返し可。既定は sleep infinity）")
    ap.add_argument("--crane", default=os.environ.get("X052_CRANE", "crane"))
    a = ap.parse_args(argv)
    ref = os.environ.get(REF_ENV, "")
    if not ref:
        print(f"{REF_ENV} が未設定です。", file=sys.stderr)
        return 2
    base = a.base or (LOCAL_BASE if a.local else PROD_BASE)
    res = verify(ref, base, a.crane, insecure=is_local_ref(ref), layer_tars=a.layer_tar,
                 base_gib=a.base_gib, limit_gib=a.limit_gib, model_name=a.model_name, cmd=a.cmd)
    for name, ok, detail in res:
        print(f"[{'合格' if ok else '不合格'}] {name}  — {detail}")
    allok = all(ok for _, ok, _ in res)
    print("総合:", "全部合格" if allok else "不合格あり")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
