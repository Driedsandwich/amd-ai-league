#!/usr/bin/env python3
"""見本の 10 問で検索を採点する（模型なし・StubBackend）。

使い方: python3 tools/eval_samples.py <starter-kit のフォルダ> [--include-images]
 - 画像が正解の問い（引用が png/jpg など）は既定で除く。
 - 「BM25 上位 8 塊」＝検索の素の上位 8 塊のファイル。
 - 「1 段目＋2 段目」＝版の見分けを当てた上位 8 塊＋識別子でたどった塊（Answerer.select）。
 - 余計なファイル＝1 段目+2 段目の塊のファイルのうち正しい引用に無い物（問ごとと合計）。
 - 採点に sample-questions.json の expected_citations を読むが、コードには埋め込まない。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))

sys.path.insert(0, HERE)
from x052 import indexer  # noqa: E402
from x052.answerer import Answerer  # noqa: E402
from x052.backend import StubBackend  # noqa: E402

IMG = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp")


class OracleStub(StubBackend):
    """--stub-oracle 用（試験の道具の中だけ）。答えを見本の期待の答えにし、sources を渡した塊の全部のファイルにする
    ＝模型が最悪に広く挙げたとき。期待の答えが空なら答えられない（拒否）と返す。"""

    def __init__(self, expected: str):
        super().__init__(expected)
        self.expected = expected

    def generate(self, prompt, max_new_tokens=128, max_time=None):
        paths = list(dict.fromkeys(re.findall(r"^### FILE: (.+?) \[", prompt, flags=re.M)))
        return json.dumps({"answerable": bool(self.expected), "answer": self.expected, "sources": paths})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kit")
    ap.add_argument("--include-images", action="store_true")
    ap.add_argument("--stub-oracle", action="store_true",
                    help="答えを期待の答えにし sources を全部のファイルにして、最終の引用・答えの一致を数える")
    args = ap.parse_args()
    corpus = os.path.join(args.kit, "mc3-corpus")
    qs = json.load(open(os.path.join(args.kit, "sample-questions.json"), encoding="utf-8"))["queries"]
    chunks, manifest = indexer.walk_corpus(corpus)
    a = Answerer(chunks, StubBackend())
    print(f"chunks={len(chunks)} manifest={len(manifest)} "
          f"(indexed={sum(m['status']=='indexed' for m in manifest)}, "
          f"skipped={sum(m['status']=='skipped' for m in manifest)}, "
          f"image={sum(m['status']=='image' for m in manifest)})")
    n = hit_plain = hit_two = extra_total = 0
    for q in qs:
        exp = set(q.get("expected_citations", []))
        if any(e.lower().endswith(IMG) for e in exp) and not args.include_images:
            print(f"Q{q['n']:>2}  skip (image)")
            continue
        plain = [chunks[i]["path"] for i in a.bm25.search(q["query"], 8)]
        sel = [chunks[i]["path"] for i in a.select(q["query"])]
        ok_plain = exp <= set(plain)
        ok_two = exp <= set(sel)
        n += 1
        extra = sorted(set(sel) - exp)
        extra_total += len(extra)
        hit_plain += ok_plain
        hit_two += ok_two
        note = "（引用が空の問い＝自明）" if not exp else ""
        print(f"Q{q['n']:>2}  expected={sorted(exp)}{note}\n      BM25上位8塊のファイル: {'OK' if ok_plain else 'NG'} {sorted(set(plain))}"
              f"\n      1段目+2段目: {'OK' if ok_two else 'NG'} {sorted(set(sel))}"
              f"\n      余計なファイル（正しい引用に無い物）: {len(extra)}")
    print(f"\nBM25 上位 8 塊: {hit_plain}/{n}   1 段目+2 段目: {hit_two}/{n}")
    if args.stub_oracle:
        from run10 import answer_matches
        cit_ok = ans_ok = both = 0
        print("\n--stub-oracle（答え＝期待・sources＝渡した塊の全部のファイル）の最終結果:")
        for q in qs:
            exp = set(q.get("expected_citations", []))
            if any(e.lower().endswith(IMG) for e in exp) and not args.include_images:
                continue
            ans, cites, _ = Answerer(chunks, OracleStub(q.get("expected_answer", ""))).answer(q["query"])
            c_ok = set(cites) == exp
            a_ok = answer_matches(ans, q.get("expected_answer", ""), q.get("answer_aliases"))
            cit_ok += c_ok
            ans_ok += a_ok
            both += c_ok and a_ok
            print(f"Q{q['n']:>2} 引用{'OK' if c_ok else 'NG'} 答え{'OK' if a_ok else 'NG'}  answer={ans!r} citations={cites}")
        print(f"最終の引用の集合が期待と完全一致: {cit_ok}/{n}   最終の答えの一致: {ans_ok}/{n}   両方: {both}/{n}")
    print(f"余計なファイルの合計（1 段目+2 段目の塊のファイルのうち正しい引用に無い物・{n} 問）: {extra_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
