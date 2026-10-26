"""依存なしの BM25。ファイル名の当たりを少し上げ、ID の 2 歩目も引く。"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set

_WORD = re.compile(r"[A-Za-z0-9]+(?:[-_.#/][A-Za-z0-9]+)*")
_CJK = re.compile(r"[぀-ヿ㐀-鿿]+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with", "what",
    "when", "where", "who", "how", "does", "do", "did", "has", "have", "had", "will", "can",
    "into", "than", "then", "there", "their", "any", "all", "each", "per",
}
# 識別子らしい語: 英大文字・数字・- の組で、英字と数字を両方含む（例 ABC-1234・X9001・PRT-77-K・REV-D4。見本の答えは書かない）
ID_RE = re.compile(r"\b(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b")


def tokenize(text: str) -> List[str]:
    """小文字化。ハイフン等でつながる語は、まるごと・部品・連結の 3 通りを出す。"""
    toks: List[str] = []
    for m in _WORD.finditer(text):
        w = m.group(0).lower()
        parts = [p for p in re.split(r"[-_.#/]", w) if p]
        if len(parts) > 1:
            toks.append(w)
            toks.append("".join(parts))
        toks.extend(parts)
    for m in _CJK.finditer(text):
        s = m.group(0)
        toks.extend(s[i:i + 2] for i in range(len(s) - 1)) if len(s) > 1 else toks.append(s)
    return toks


def query_terms(query: str) -> List[str]:
    seen, out = set(), []
    for t in tokenize(query):
        if t in STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


class BM25:
    def __init__(self, chunks: Sequence[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.post: Dict[str, List] = defaultdict(list)
        self.dl: List[int] = []
        self.path_toks: List[Set[str]] = []
        for i, c in enumerate(self.chunks):
            toks = tokenize(c["text"])
            self.dl.append(len(toks))
            for t, n in Counter(toks).items():
                self.post[t].append((i, n))
            self.path_toks.append(set(tokenize(c["path"])))
        self.n = len(self.chunks)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0

    def idf(self, t: str) -> float:
        df = len(self.post.get(t, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str, path_bonus: float = 1.0) -> Dict[int, float]:
        terms = query_terms(query)
        sc: Dict[int, float] = defaultdict(float)
        for t in terms:
            idf = self.idf(t)
            for i, tf in self.post.get(t, ()):
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                sc[i] += idf * tf * (self.k1 + 1) / denom
        if path_bonus and sc:
            # 問いの語がファイル名に当たる塊を少し上げる（長い語・珍しい語ほど強く）
            top = max(sc.values())
            for i in range(self.n):
                hit = [t for t in terms if t in self.path_toks[i] and len(t) > 2]
                if hit:
                    sc[i] += path_bonus * 0.15 * top * min(len(hit), 3)
        return sc

    def search(self, query: str, k: int = 8, path_bonus: float = 1.0) -> List[int]:
        sc = self.scores(query, path_bonus)
        order = sorted(sc, key=lambda i: (-sc[i], i))
        return order[:k]


def build_id_table(chunks: Sequence[dict]) -> Dict[str, List[int]]:
    """識別子 → それを含む塊の添字の一覧（INDEX_DIR/ids.json に保存する形）。"""
    table: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        for ident in dict.fromkeys(ID_RE.findall(c["text"])):
            table.setdefault(ident, []).append(i)
    return table


HOP_MIN_SCORE = 0.3
HOP_PER_FILE = 2


def second_hop(chunks: Sequence[dict], table: Dict[str, List[int]], first: Sequence[int],
               query: str, max_add: int = 4, per_file: int = HOP_PER_FILE,
               scores: Optional[Dict[int, float]] = None) -> List[tuple]:
    """識別子でたどる 2 段目（1 回だけ・足すのは max_add 塊まで・ファイルごとに per_file 塊まで）。

    1 段目の塊に出る識別子のうち、問いに無く、2 つ以上のファイルに出る物を点で選ぶ:
      ① 問いに無い（問いにある識別子は対象外）
      ② 出るファイルが少ないほど高い（rarity = 1/(ファイル数-1)）
      ③ 足す候補の塊が問いと BM25 で近いほど高い（closeness = その塊の点/最高点）
    点 = 0.6*② + 0.4*③。HOP_MIN_SCORE 未満は捨てる。ファイル数の上限で一律には切らない。
    返り値: [(足す塊の添字, 識別子, 起点のファイル)]。足す塊は 1 段目に入っていない物。
    """
    qlow = query.lower()
    have = set(first)
    top = max(scores.values()) if scores else 0.0

    def closeness(j: int) -> float:
        return (scores.get(j, 0.0) / top) if top > 0 else 0.5

    cands: List[tuple] = []  # (点, 出現順, 識別子, 起点, 足す候補の塊 [近い順])
    seen_ids = set()
    for i in first:
        origin = chunks[i]["path"]
        for ident in dict.fromkeys(ID_RE.findall(chunks[i]["text"])):
            if ident in seen_ids or ident.lower() in qlow:
                continue
            seen_ids.add(ident)
            hits = table.get(ident, [])
            files = {chunks[j]["path"] for j in hits}
            if len(files) < 2:
                continue
            pool = [j for j in hits if chunks[j]["path"] != origin and j not in have]
            if not pool:
                continue
            pool.sort(key=lambda j: (-closeness(j), j))
            score = 0.6 / (len(files) - 1) + 0.4 * closeness(pool[0])
            if score < HOP_MIN_SCORE:
                continue
            cands.append((score, len(cands), ident, origin, pool))
    cands.sort(key=lambda t: (-t[0], t[1]))
    adds: List[tuple] = []
    per: Dict[str, int] = {}
    for _, _, ident, origin, pool in cands:
        for j in pool:
            if len(adds) >= max_add:
                return adds
            path = chunks[j]["path"]
            if j in have or per.get(path, 0) >= per_file:
                continue
            adds.append((j, ident, origin))
            have.add(j)
            per[path] = per.get(path, 0) + 1
    return adds
