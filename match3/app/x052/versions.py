"""版の見分け。索引の全体で版の組を作り、旧版・撤回を下げ、現行の版へ振り替える。

撤回の根拠は 2 つだけ:
  (1) ファイル名・置き場の印（WITHDRAWN・old・draft・superseded・archive など）
  (2) 本文の頭（先頭 500 字）にある「この文書自身」の状態の言明。
本文で別の版（自分と違う版の番号）を撤回と述べている物は、相手を旧版にする（自分は新版）。
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

VERSION_TOKEN = re.compile(r"^(?:r|v|rev)\d+[a-z]?$|^(?:rev|version)$", re.I)
STATUS_WORDS = {"withdrawn", "old", "draft", "superseded", "archive", "archived", "obsolete", "deprecated"}
HEAD_CHARS = 500
_NUM = re.compile(r"(?:^|[_\-. ])(?:r|v|rev)(\d+)", re.I)
_WD = re.compile(r"\b(?:withdrawn|superseded|obsolete|obsoleted|retired|deprecated|no longer (?:valid|current)|"
                 r"do not use)\b|撤回|廃止|無効", re.I)
_ACTIVE = re.compile(r"\b(?:supersedes|replaces|obsoletes|retires)\b", re.I)
_BY = re.compile(r"\b(?:superseded|replaced|obsoleted|withdrawn|retired)\s+by\b", re.I)
_VERNUM = re.compile(r"\b(?:rev(?:ision)?|r|v|version)\s*\.?\s*(\d+)\b", re.I)


def split_name(path: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    return [t for t in re.split(r"[_\-. ]+", stem) if t]


def base_key(path: str) -> Tuple[str, Tuple[str, ...]]:
    """版の語を除いた名前。同じなら同じ文書の版違い。"""
    d = os.path.dirname(path)
    toks = [t.lower() for t in split_name(path) if not VERSION_TOKEN.match(t) and t.lower() not in STATUS_WORDS]
    return d, tuple(toks)


def name_withdrawn(path: str) -> bool:
    parts = [p.lower() for p in re.split(r"[/\\]", os.path.dirname(path))]
    return any(t.lower() in STATUS_WORDS for t in split_name(path)) or any(p in STATUS_WORDS for p in parts)


def version_number(path: str) -> int:
    m = _NUM.search(os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else 0


def head_statements(head: str, own: int) -> Tuple[bool, Set[int], bool]:
    """本文の頭から (自分が撤回か, 旧版と述べられた別の版の番号, 「前の版」を置き換えると述べたか)。"""
    self_old, other_old, prev_old = False, set(), False
    for sent in re.split(r"[.\n;。]+", head[:HEAD_CHARS]):
        has_wd = bool(_WD.search(sent))
        active = bool(_ACTIVE.search(sent))
        if not (has_wd or active):
            continue
        if _BY.search(sent):
            self_old = True  # 「…により置き換えられた」＝自分が旧版
            continue
        nums = {int(n) for n in _VERNUM.findall(sent)}
        if nums:
            for n in nums:
                if own and n == own:
                    self_old = self_old or has_wd
                else:
                    other_old.add(n)
        elif active:
            prev_old = True
        else:
            self_old = True
    return self_old, other_old, prev_old


class VersionTable:
    """索引の全体（全部の塊）から版の組を作る。"""

    def __init__(self, chunks: Sequence[dict]):
        first_text: Dict[str, str] = {}
        for c in chunks:
            first_text.setdefault(c["path"], c["text"])
        self.groups: Dict[Tuple, List[str]] = {}
        for p in first_text:
            self.groups.setdefault(base_key(p), []).append(p)
        self.self_old: Dict[str, bool] = {}
        self.old: Set[str] = set()
        self.current_of: Dict[str, str] = {}
        for key, ps in self.groups.items():
            info = {p: head_statements(first_text[p], version_number(p)) for p in ps}
            for p in ps:
                self.self_old[p] = name_withdrawn(p) or info[p][0]
            if len(ps) < 2:
                continue
            for p in ps:
                _, others, prev = info[p]
                vp = version_number(p)
                for q in ps:
                    vq = version_number(q)
                    if q != p and ((vq in others and vq != vp) or (prev and vq < vp)):
                        self.self_old[q] = True
            ranked = sorted(ps, key=lambda p: (not self.self_old[p], version_number(p), p), reverse=True)
            cur = ranked[0]
            for p in ps:
                if p != cur:
                    self.old.add(p)
                    self.current_of[p] = cur
        for p, v in self.self_old.items():
            if v:
                self.old.add(p)  # 相手の無い撤回（置き場の印だけ）は下げるだけ

    def is_old(self, path: str) -> bool:
        return path in self.old

    def current(self, path: str) -> Optional[str]:
        return self.current_of.get(path)


def rerank(chunks: Sequence[dict], order: Sequence[int], table: Optional[VersionTable] = None,
           ) -> Tuple[List[int], Dict[str, str]]:
    """候補の順番（塊の添字）に版の見分けを当てる。

    返り値: (順番, {引かれた旧版のパス: 現行の版のパス})
    現行の版の組があるファイルの旧版の塊は外す。相手の無い撤回は後ろへ下げる。
    現行の版の塊が候補に無いときの補充は呼び出し側（Answerer）が replaced を見て行う。
    """
    t = table or VersionTable(chunks)
    replaced: Dict[str, str] = {}
    kept: List[int] = []
    for i in order:
        p = chunks[i]["path"]
        cur = t.current(p)
        if cur is not None and cur != p:
            replaced[p] = cur
            continue
        kept.append(i)
    good = [i for i in kept if chunks[i]["path"] not in t.old]
    bad = [i for i in kept if chunks[i]["path"] in t.old]
    return good + bad, replaced
