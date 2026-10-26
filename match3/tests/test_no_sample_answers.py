"""提出するコード（app/）に見本の答えと引用のファイル名が入っていないかを見る。

主催の約束: "A solution that hardcodes an answer from this sample scores zero."
見本の答えは X052_KIT の sample-questions.json から読む（コードには書かない）。
"""

import json
import os
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
KIT = os.environ.get("X052_KIT")


def _needles(kit: str) -> list[str]:
    qs = json.load(open(os.path.join(kit, "sample-questions.json"), encoding="utf-8"))["queries"]
    out = set()
    for q in qs:
        # 数だけの短い答え（"94" など）は一般の数と見分けられないので除き、英字か点を含む物を見る
        for a in [q.get("expected_answer", "")] + list(q.get("answer_aliases", [])):
            if len(a) >= 3 and re.search(r"[A-Za-z.]", a) and not re.fullmatch(r"[\d\s]+[A-Za-z°]*s?", a):
                out.add(a)
        for c in q.get("expected_citations", []):
            out.add(Path(c).stem)
    return sorted(out)


def _scan(root: Path, needles: list[str]) -> list[tuple[str, str]]:
    hits = []
    for p in sorted(root.rglob("*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for n in needles:
            if n in text:
                hits.append((str(p.relative_to(root)), n))
    return hits


@pytest.mark.skipif(not KIT, reason="X052_KIT not set")
def test_app_has_no_sample_answers():
    needles = _needles(KIT)
    assert len(needles) >= 10, needles
    assert _scan(APP, needles) == []


@pytest.mark.skipif(not KIT, reason="X052_KIT not set")
def test_scan_detects_a_planted_answer(tmp_path):
    needles = _needles(KIT)
    planted = tmp_path / "planted.py"
    planted.write_text(f"PROMPT = 'for example {needles[0]}'\n", encoding="utf-8")
    assert _scan(tmp_path, needles) == [("planted.py", needles[0])]
