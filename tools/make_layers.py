#!/usr/bin/env python3
"""層のファイル（tar）を作る。同じ入力なら同じ tar（持ち主 0/0・時刻固定・パス順・Python の tarfile）。

  app の層    : --app-dir  → /app/（空の /app/corpus /app/output /app/index を含む）
  依存の層    : --site-dir → /opt/x052/site/
  重みの層    : --weights-dir → /models/（ファイル単位で --weights-max-gib 未満ずつ複数の層に分ける）
各層の非圧縮の大きさ（中のファイルの合計）と、土台（既定 29.2GiB）を足した合計が 60GiB に収まるかを出す。
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

GIB = 1024 ** 3
FIXED_MTIME = 315532800  # 1980-01-01
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache"}
EXCLUDE_FILES = {".DS_Store"}


def _info(name: str, kind: str, mode: int, size: int = 0, linkname: str = "") -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type = {"dir": tarfile.DIRTYPE, "file": tarfile.REGTYPE, "link": tarfile.SYMTYPE}[kind]
    ti.mode, ti.size, ti.linkname = mode, size, linkname
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = FIXED_MTIME
    return ti


def _walk(src: Path, exclude: bool) -> List[Tuple[str, Path, str]]:
    """(src からの相対パス, 実パス, 種類) をパス順で返す。"""
    out: List[Tuple[str, Path, str]] = []

    def rec(d: Path, rel: str):
        for name in sorted(os.listdir(d)):
            p = d / name
            r = f"{rel}/{name}" if rel else name
            if p.is_symlink():
                out.append((r, p, "link"))
            elif p.is_dir():
                if exclude and name in EXCLUDE_DIRS:
                    continue
                out.append((r, p, "dir"))
                rec(p, r)
            elif p.is_file():
                if exclude and (name in EXCLUDE_FILES or name.endswith(".pyc")):
                    continue
                out.append((r, p, "file"))

    rec(src, "")
    return out


def _parents(dest: str) -> List[str]:
    parts = dest.strip("/").split("/")
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def write_tar(tar_path: Path, dest: str, entries: List[Tuple[str, Path, str]],
              extra_dirs: Iterable[str] = (), with_parents: bool = True) -> int:
    """dest（例 'opt/x052/site'）の下に entries を置く tar を書く。中のファイルの合計（byte）を返す。"""
    dest = dest.strip("/")
    names = {}
    if with_parents:
        for d in _parents(dest):
            names[d] = ("dir", None)
    for d in extra_dirs:
        names[f"{dest}/{d}"] = ("dir", None)
    for rel, p, kind in entries:
        names[f"{dest}/{rel}"] = (kind, p)
    total = 0
    with open(tar_path, "wb") as raw, tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in sorted(names):
            kind, p = names[name]
            if kind == "dir":
                tf.addfile(_info(name, "dir", 0o755))
            elif kind == "link":
                tf.addfile(_info(name, "link", 0o777, linkname=os.readlink(p)))
            else:
                st = p.stat()
                mode = 0o755 if st.st_mode & stat.S_IXUSR else 0o644
                with open(p, "rb") as f:
                    tf.addfile(_info(name, "file", mode, st.st_size), f)
                total += st.st_size
    return total


def split_by_size(files: List[Tuple[str, Path, str]], max_bytes: int) -> List[List[Tuple[str, Path, str]]]:
    """ファイル単位で max_bytes 未満ずつの組に分ける（パス順を保つ）。1 ファイルで超えるなら例外。"""
    groups: List[List] = [[]]
    cur = 0
    for e in files:
        size = e[1].stat().st_size if e[2] == "file" else 0
        if size >= max_bytes:
            raise ValueError(f"{e[0]} is {size} bytes; cannot fit under {max_bytes} (a single file cannot be split)")
        if cur + size >= max_bytes and groups[-1]:
            groups.append([])
            cur = 0
        groups[-1].append(e)
        cur += size
    return groups


def make_layers(out_dir: Path, app_dir: Optional[Path] = None, site_dir: Optional[Path] = None,
                weights_dir: Optional[Path] = None, weights_max_bytes: int = 9 * GIB - 1,
                base_gib: float = 29.2, limit_gib: float = 60.0) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    layers: List[Dict] = []
    if app_dir:
        p = out_dir / "layer_app.tar"
        n = write_tar(p, "app", _walk(app_dir, True), extra_dirs=("corpus", "output", "index"))
        layers.append({"name": "app", "path": str(p), "uncompressed_bytes": n})
    if site_dir:
        p = out_dir / "layer_site.tar"
        n = write_tar(p, "opt/x052/site", _walk(site_dir, False))
        layers.append({"name": "site", "path": str(p), "uncompressed_bytes": n})
    if weights_dir:
        files = [e for e in _walk(weights_dir, False) if e[2] != "dir"]
        for i, grp in enumerate(split_by_size(files, weights_max_bytes), 1):
            p = out_dir / f"layer_weights_{i:02d}.tar"
            n = write_tar(p, "models", grp, with_parents=(i == 1))
            layers.append({"name": f"weights_{i:02d}", "path": str(p), "uncompressed_bytes": n})
    own = sum(l["uncompressed_bytes"] for l in layers)
    total = own + int(base_gib * GIB)
    report = {"layers": layers, "own_bytes": own, "base_gib": base_gib,
              "total_gib": total / GIB, "limit_gib": limit_gib,
              "headroom_gib": limit_gib - total / GIB, "fits": total <= limit_gib * GIB}
    (out_dir / "layers.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--app-dir", type=Path)
    ap.add_argument("--site-dir", type=Path)
    ap.add_argument("--weights-dir", type=Path)
    ap.add_argument("--weights-max-gib", type=float, default=9.0)
    ap.add_argument("--base-gib", type=float, default=29.2)
    ap.add_argument("--limit-gib", type=float, default=60.0)
    a = ap.parse_args(argv)
    r = make_layers(a.out_dir, a.app_dir, a.site_dir, a.weights_dir,
                    weights_max_bytes=int(a.weights_max_gib * GIB) - 1,
                    base_gib=a.base_gib, limit_gib=a.limit_gib)
    for l in r["layers"]:
        print(f"{l['name']:<12} {l['uncompressed_bytes'] / GIB:8.3f} GiB  {l['path']}")
    print(f"自分の層の合計 {r['own_bytes'] / GIB:.3f} GiB + 土台 {r['base_gib']} GiB = {r['total_gib']:.3f} GiB"
          f"（上限 {r['limit_gib']} GiB・余裕 {r['headroom_gib']:.3f} GiB・{'収まる' if r['fits'] else '超える'}）")
    return 0 if r["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
