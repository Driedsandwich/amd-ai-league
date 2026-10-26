"""模型の返りを、主催の書き方の決まりに寄せる。

- 前後の空白・囲みの引用符・バッククォート・「Text:」などの前置き・文末の句点を落とす。
- 改行と連続の空白を 1 つの空白にする。
- ナンバープレートらしい語があるときだけ、米国の州の名前を独立した語として落とす（標識の語は落とさない）。
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# 2 語の州名を先に・1 語を後に（最長一致）。
US_STATES_2W = [
    "NEW YORK", "NEW JERSEY", "NEW MEXICO", "NEW HAMPSHIRE", "NORTH CAROLINA", "NORTH DAKOTA",
    "SOUTH CAROLINA", "SOUTH DAKOTA", "WEST VIRGINIA", "RHODE ISLAND",
]
US_STATES_1W = [
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", "CONNECTICUT", "DELAWARE",
    "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY",
    "LOUISIANA", "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI",
    "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WISCONSIN", "WYOMING",
]
US_STATES = US_STATES_2W + US_STATES_1W

PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:transcription|transcribed\s+text|text|answer|output|result|ocr|"
    r"licen[sc]e\s+plate(?:\s+number)?|plate\s+number|plate|sign\s+text|response)\s*(?:is\s*)?[:：\-]\s*",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")
QUOTE_PAIRS = [('"', '"'), ("'", "'"), ("`", "`"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』")]
CJK_RE = re.compile(r"[㐀-鿿]")
TRAILING_PERIODS = ".。．"


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    s = FENCE_RE.sub("", s).strip()
    for _ in range(3):  # 入れ子の囲みや前置きの後の囲みも落とす
        before = s
        s = PREFIX_RE.sub("", s).strip()
        for a, b in QUOTE_PAIRS:
            if len(s) >= 2 and s[0] == a and s[-1] == b:
                s = s[1:-1].strip()
        if s == before:
            break
    return s


def _is_mixed(tok: str) -> bool:
    t = re.sub(r"[-.·_]", "", tok)
    return 4 <= len(t) <= 8 and bool(re.search(r"[A-Za-z]", t)) and bool(re.search(r"\d", t))


def _state_spans(words: List[str]) -> List[Tuple[int, int]]:
    """州名に当たる語の範囲 [i, j) の一覧（重ならない・最長一致）。"""
    up = [w.upper().strip(",;:") for w in words]
    spans, i = [], 0
    while i < len(up):
        hit: Optional[int] = None
        if i + 1 < len(up) and f"{up[i]} {up[i + 1]}" in US_STATES_2W:
            hit = 2
        elif up[i] in US_STATES_1W:
            hit = 1
        if hit:
            spans.append((i, i + hit))
            i += hit
        else:
            i += 1
    return spans


def looks_like_plate(words: List[str], skip: set) -> bool:
    """英字と数字が混ざる 4〜8 字の語、英字の語＋数字の語の並び（QWT 4402 型。間があいたプレート）、または漢字で始まる番号。"""
    rest = [(i, w) for i, w in enumerate(words) if i not in skip]
    if any(_is_mixed(w) for _, w in rest):
        return True
    if rest and CJK_RE.match(rest[0][1]) and len(rest) >= 2:
        return True
    for (i, a), (j, b) in zip(rest, rest[1:]):
        if j == i + 1 and re.fullmatch(r"[A-Za-z]{1,4}", a) and re.fullmatch(r"\d{2,5}", b) and len(a) + len(b) <= 8:
            return True
    return False


def drop_state_names(text: str) -> str:
    words = text.split(" ")
    spans = _state_spans(words)
    if not spans:
        return text
    skip = {k for a, b in spans for k in range(a, b)}
    if not looks_like_plate(words, skip):
        return text
    kept = [w for i, w in enumerate(words) if i not in skip]
    return " ".join(kept) if kept else text


def clean(raw: Optional[str]) -> str:
    if not raw:
        return ""
    s = _strip_wrappers(str(raw))
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(TRAILING_PERIODS).strip()
    s = _strip_wrappers(s)
    s = re.sub(r"\s+", " ", s).strip()
    return drop_state_names(s) if s else ""
