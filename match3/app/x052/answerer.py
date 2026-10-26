"""問い→候補→模型→答えの持ち主の確認→引用の絞り→正規化。

方針（監査の直し）:
- 模型の返りは型を検査し、満たす物だけ採る（answerable=真偽値・answer=文字列・sources=文字列の配列）。
- 答えの値を含む塊（持ち主）が無い答えは出さない（根拠の無い答えを出さない）。
- 引用＝持ち主のファイル 1 つ＋（2 段目でたどったとき等）たどりの起点のファイル。模型の sources は持ち主を選ぶ優先に使うだけ。
- 長い答えからは、問いの値の型に合い、持ち主の塊に現れる値だけを取る（語の数では分けない）。
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from . import versions
from .retrieve import BM25, ID_RE, build_id_table, second_hop

TOP_K = 8
RETRIEVE_K = 16
MAX_NEW_TOKENS = 160
SUPPLEMENT_CHUNKS = 2
GEN_MARGIN_S = 1.5  # 生成の max_time は残り時間からこれを引く

RULES = """You answer questions using ONLY the excerpts below. The products and companies are fictional, so never use outside knowledge.
Rules:
- Give the VALUE only (a number, part number, version, quarter, code), not a sentence. Keep qualifiers that are part of the value, e.g. "Q1 FY29" rather than "Q1", "REV-D4" rather than "D4". Do not add units unless they are part of the value as written.
- If an excerpt says a document is WITHDRAWN or superseded, or a file name shows an older revision, prefer the current document.
- If the excerpts do not contain the answer, set answerable to false and leave answer empty. Do not guess.
- In "sources" list ONLY the files (exact paths from the FILE lines) whose removal would make the answer impossible. If one file is enough, list one. If two files are both needed (for example one gives a ticket id and the other gives the fixed version), list both.
Reply with one JSON object and nothing else:
{"answerable": true or false, "answer": "<value>", "sources": ["<path>", ...]}
"""

# 問いの言葉 → 値の型（先に当たった物）。(名前, 問いの言葉, 値の形)
_VALUE_TYPES = [
    ("quarter", re.compile(r"\bquarters?\b", re.I), re.compile(r"\bQ[1-4]\s?FY\s?\d{2,4}\b", re.I)),
    ("revision", re.compile(r"\brevisions?\b|\brev\b", re.I), re.compile(r"\bREV-?[A-Z0-9]+\b")),
    ("pin", re.compile(r"\bpins?\b", re.I), re.compile(r"\b[A-Z]{1,2}\d{1,3}\b")),
    ("errcode", re.compile(r"\b(?:error|fault|status|exit|return) codes?\b|\bcodes?\b", re.I),
     re.compile(r"\b[A-Z]{1,3}-?\d{3,}\b")),
    ("part", re.compile(r"\bpart (?:number|no)\b|\bmodel (?:number|no)\b|\bserial\b|\bsku\b|\bidentifier\b", re.I),
     re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")),
    ("version", re.compile(r"\b(?:version|firmware|release|build)\b", re.I),
     re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")),
    ("number", re.compile(r"\b(?:temperature|timeout|seconds|minutes|hours|price|cost|how many|number of|maximum|"
                          r"minimum|limit|capacity|count|rate|power|voltage|speed|size|length|default|percent|"
                          r"watts?|ms)\b", re.I),
     re.compile(r"(?<![A-Za-z0-9-])-?\$?\d[\d,]*(?:\.\d+)?")),
]


def value_pattern(query: str) -> Optional["re.Pattern"]:
    for _, trigger, shape in _VALUE_TYPES:
        if trigger.search(query):
            return shape
    return None


def normalize_answer(ans: str) -> str:
    a = (ans or "").strip()
    a = re.sub(r"^(?:the\s+)?(?:answer|value)\s*(?:is|:)\s*", "", a, flags=re.I).strip()
    a = a.rstrip("。.．").strip().strip("\"'`").strip()
    return a.rstrip("。.．").strip()


_PAT_CACHE: Dict[str, "re.Pattern"] = {}


def _value_regex(v: str) -> "re.Pattern":
    pat = _PAT_CACHE.get(v)
    if pat is None:
        chars = [c for c in v if c.isalnum()]
        if chars:
            body = r"[\W_]*".join(re.escape(c) for c in chars)
            pat = re.compile(r"(?<![0-9A-Za-z])(?<!\d[.,])" + body + r"(?![0-9A-Za-z])(?![.,]\d)", re.I)
        else:
            pat = re.compile(re.escape(v))
        _PAT_CACHE[v] = pat
    return pat


def contains_value(text: str, v: str) -> bool:
    """値 v を含むか。空白と記号の違いは無視し、数字や英字の途中（1947 の中の 94 など）には当てない。"""
    return bool(v) and bool(_value_regex(v).search(text))


def parse_model_json(raw: str) -> Optional[dict]:
    """型を検査して、満たす最初のオブジェクトだけ返す。無ければ None。"""
    if not raw:
        return None
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M)
    dec = json.JSONDecoder()
    pos = s.find("{")
    while pos != -1:
        try:
            obj, end = dec.raw_decode(s, pos)
        except ValueError:
            pos = s.find("{", pos + 1)
            continue
        if (isinstance(obj, dict) and isinstance(obj.get("answerable"), bool)
                and isinstance(obj.get("answer"), str) and isinstance(obj.get("sources"), list)
                and all(isinstance(x, str) for x in obj["sources"])):
            return obj
        pos = s.find("{", pos + 1)
    return None


class Answerer:
    def __init__(self, chunks: Sequence[dict], backend, id_table: Optional[Dict[str, List[int]]] = None):
        self.chunks = list(chunks)
        self.bm25 = BM25(self.chunks)
        self.id_table = id_table if id_table is not None else build_id_table(self.chunks)
        self.vtable = versions.VersionTable(self.chunks)  # 版の組は索引の全体で作る
        self.file_chunks: Dict[str, List[int]] = {}
        for i, c in enumerate(self.chunks):
            self.file_chunks.setdefault(c["path"], []).append(i)
        self.backend = backend
        self.links: Dict[int, Tuple[str, str]] = {}
        self._scores: Dict[int, float] = {}

    # ---- 候補 ----
    def _supplement(self, out: List[int], replaced: Dict[str, str]) -> List[int]:
        """引かれた旧版に対し、現行の版の塊が候補に無ければ、問いに近い順に足す。"""
        have_files = {self.chunks[i]["path"] for i in out}
        for cur in dict.fromkeys(replaced.values()):
            if cur in have_files:
                continue
            ids = sorted(self.file_chunks.get(cur, []), key=lambda i: (-self._scores.get(i, 0.0), i))
            out = out + ids[:SUPPLEMENT_CHUNKS]
        return out

    def select(self, query: str) -> List[int]:
        """1 段目: BM25→版の見分け→上位 TOP_K。2 段目: 識別子で 4 塊まで足す（1 回だけ）。全体へ版の見分け。"""
        self._scores = self.bm25.scores(query)
        base = sorted(self._scores, key=lambda i: (-self._scores[i], i))[:RETRIEVE_K]
        ranked, replaced = versions.rerank(self.chunks, base, self.vtable)
        first = self._supplement(ranked[:TOP_K], replaced)
        adds = second_hop(self.chunks, self.id_table, first, query, scores=self._scores)
        out, replaced2 = versions.rerank(self.chunks, first + [j for j, _, _ in adds], self.vtable)
        out = self._supplement(out, {**replaced, **replaced2})
        keep = set(out)
        self.links = {j: (ident, origin) for j, ident, origin in adds if j in keep}
        return out

    def build_prompt(self, query: str, idx: Sequence[int]) -> str:
        parts = [RULES, "\nExcerpts:\n"]
        for i in idx:
            c = self.chunks[i]
            note = ""
            if i in self.links:
                note = f"(linked by identifier {self.links[i][0]} found in {self.links[i][1]})\n"
            parts.append(f"### FILE: {c['path']} [{c['loc']}]\n{note}{c['text']}\n")
        parts.append(f"\nQuestion: {query}\nJSON:")
        return "\n".join(parts)

    # ---- 答え・持ち主・引用 ----
    def resolve_value(self, answer: str, query: str, idx: Sequence[int]) -> Optional[Tuple[str, List[int]]]:
        """(値, 持ち主の塊) を返す。値を含む塊が無ければ None。長い答えは型に合う値だけを取る。"""
        pat = value_pattern(query)
        cands = [answer]
        if pat is not None and not pat.fullmatch(answer):
            cands += list(dict.fromkeys(m.group(0).strip().rstrip("。.,") for m in pat.finditer(answer)))
        for v in cands:
            owners = [i for i in idx if contains_value(self.chunks[i]["text"], v)]
            if owners:
                return v, owners
        return None

    def choose_owner(self, owners: Sequence[int], sources: Sequence[str]) -> int:
        """持ち主の塊を 1 つ選ぶ: 模型の sources にあるファイル優先→渡した順。旧版なら現行の版の持ち主へ。"""
        paths = list(dict.fromkeys(self.chunks[i]["path"] for i in owners))
        def prio(p):
            return (0 if p in sources else 1, paths.index(p))
        best = sorted(paths, key=prio)[0]
        cur = self.vtable.current(best)
        if cur and cur in paths:
            best = cur
        cand = [i for i in owners if self.chunks[i]["path"] == best]
        # 同じファイルの中では、2 段目でたどった塊（識別子が指す行）を先に、次に問いに近い塊
        return max(cand, key=lambda i: (i in self.links, self._scores.get(i, 0.0), -i))

    def origin_file(self, owner: int, query: str, idx: Sequence[int]) -> Optional[str]:
        """たどりの起点のファイル。2 段目でたどった塊なら、その起点。そうでなくても、持ち主の塊の識別子
        （問いに無い）が、問いにより近い別ファイルの塊にもあれば、そのファイル（持ち主は起点から引いた行）。"""
        if owner in self.links:
            return self.links[owner][1]
        qlow = query.lower()
        owner_path = self.chunks[owner]["path"]
        mine = {x for x in ID_RE.findall(self.chunks[owner]["text"]) if x.lower() not in qlow}
        if not mine:
            return None
        best, best_score = None, self._scores.get(owner, 0.0)
        for i in idx:
            c = self.chunks[i]
            if c["path"] == owner_path or self.vtable.is_old(c["path"]):
                continue
            if mine & set(ID_RE.findall(c["text"])) and self._scores.get(i, 0.0) > best_score:
                best, best_score = c["path"], self._scores.get(i, 0.0)
        return best

    def answer(self, query: str, deadline: Optional[float] = None) -> Tuple[str, List[str], float]:
        """deadline は time.monotonic() の絶対の時刻（None なら無期限）。"""
        empty = ("", [], 0.0)
        if not self.chunks or not query.strip():
            return empty
        max_time = None
        if deadline is not None:
            max_time = deadline - time.monotonic() - GEN_MARGIN_S
            if max_time < 0.3:
                return empty
        idx = self.select(query)
        if not idx:
            return empty
        raw = self.backend.generate(self.build_prompt(query, idx), MAX_NEW_TOKENS, max_time=max_time)
        obj = parse_model_json(raw)
        if obj is None or not obj["answerable"]:
            return empty
        ans = normalize_answer(obj["answer"])
        if not ans:
            return empty
        got = self.resolve_value(ans, query, idx)
        if got is None:
            return empty  # 根拠の塊が無い答えは出さない
        value, owners = got
        sources = [s[2:] if s.startswith("./") else s for s in obj["sources"]]
        owner = self.choose_owner(owners, sources)
        cites = [self.chunks[owner]["path"]]
        origin = self.origin_file(owner, query, idx)
        if origin and origin not in cites:
            cites.append(origin)
        return value, cites, 0.9
