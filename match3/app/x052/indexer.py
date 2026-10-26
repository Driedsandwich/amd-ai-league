"""束を歩いて塊と目録を作り保存する。1 ファイルごとに例外を捕まえて続ける。"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from . import parsers
from .retrieve import build_id_table


def _reason(exc: BaseException) -> Tuple[str, str]:
    if isinstance(exc, parsers.EncryptedError):
        return "skipped", "encrypted"
    if isinstance(exc, parsers.UnknownTypeError):
        return "skipped", "unknown binary type"
    if isinstance(exc, parsers.EmptyFileError):
        return "skipped", f"empty: {exc}"
    if isinstance(exc, PermissionError):
        return "skipped", f"unreadable: PermissionError ({exc.strerror or exc})"
    if isinstance(exc, OSError):
        return "skipped", f"unreadable: {type(exc).__name__} ({exc.strerror or exc})"
    return "skipped", f"parse error: {type(exc).__name__}: {str(exc)[:200]}"


def _rel(corpus: str, path: str) -> str:
    return os.path.relpath(path, corpus).replace(os.sep, "/")


def walk_corpus(corpus: str, parse_budget: Optional[float] = None) -> Tuple[List[dict], List[dict]]:
    """(塊, 目録) を返す。ここでは何があっても例外を外へ出さない。

    parse_budget（秒）を過ぎたら、残りのファイルは読まずに目録へ skipped_time と書いて先へ進む。
    読んでいる途中で過ぎたファイルは、読めた分だけ入れて truncated: true と書く。
    """
    corpus = str(corpus)
    deadline = None if parse_budget is None else time.monotonic() + parse_budget
    chunks: List[dict] = []
    manifest: List[dict] = []

    def onerror(err: OSError):
        p = getattr(err, "filename", None) or corpus
        manifest.append({"path": _rel(corpus, p) if p else ".", "status": "skipped",
                         "kind": "dir", "reason": f"unreadable directory: {type(err).__name__}"})

    try:
        walker = list(os.walk(corpus, onerror=onerror))
    except Exception as e:  # 想定外でも空で返す
        manifest.append({"path": ".", "status": "skipped", "kind": "dir", "reason": f"walk failed: {e}"})
        return chunks, manifest

    walker.sort(key=lambda t: t[0])
    for dirpath, dirnames, filenames in walker:
        dirnames.sort()
        filenames.sort()
        if not dirnames and not filenames and dirpath != corpus:
            manifest.append({"path": _rel(corpus, dirpath), "status": "skipped", "kind": "dir",
                             "reason": "empty directory"})
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = _rel(corpus, full)
            kind = parsers.kind_of(full)
            if deadline is not None and time.monotonic() > deadline:
                manifest.append({"path": rel, "status": "skipped", "kind": kind, "reason": "skipped_time"})
                continue
            try:
                segs = parsers.parse_file(full, deadline)
            except parsers.ImageFile:
                manifest.append({"path": rel, "status": "image", "kind": "image", "reason": "needs vision"})
                continue
            except BaseException as exc:  # noqa: BLE001 - 1 ファイルの失敗で止まらない
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                status, reason = _reason(exc)
                manifest.append({"path": rel, "status": status, "kind": kind, "reason": reason})
                continue
            n = 0
            for loc, text in segs:
                if not text.strip():
                    continue
                chunks.append({"path": rel, "loc": loc, "text": text})
                n += 1
            entry = {"path": rel, "status": "indexed", "kind": kind, "chunks": n}
            if getattr(segs, "truncated", False):
                entry["truncated"] = True
                entry["truncated_reason"] = segs.truncated_reason
            manifest.append(entry)
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks, manifest


def save_index(index_dir: str, chunks: List[dict], manifest: List[dict]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    tmp = os.path.join(index_dir, "chunks.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(index_dir, "chunks.jsonl"))
    tmp = os.path.join(index_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(index_dir, "manifest.json"))
    tmp = os.path.join(index_dir, "ids.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(build_id_table(chunks), f, ensure_ascii=False)
    os.replace(tmp, os.path.join(index_dir, "ids.json"))


def load_chunks(index_dir: str, name: str = "chunks.jsonl") -> List[dict]:
    p = os.path.join(index_dir, name)
    out: List[dict] = []
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_index(corpus: str, index_dir: str, parse_budget: Optional[float] = None) -> List[dict]:
    chunks, manifest = walk_corpus(corpus, parse_budget)
    save_index(index_dir, chunks, manifest)
    return manifest
