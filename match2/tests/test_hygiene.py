"""app/ の中の衛生: 外へ出る通信が無い・見本の答えが入っていない。

見本の答えは tools/samples.json から読む（コードには書かない）。主催の約束: "A solution that hardcodes answers
to the samples scores zero."
"""
import json
import re
from pathlib import Path

import pytest

from conftest import APP, TOOLS

SAMPLES = json.loads((TOOLS / "samples.json").read_text(encoding="utf-8"))["samples"]


def app_files():
    return sorted(p for p in APP.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


# ---- 外へ出る通信が無い ----
NET_NEEDLES = [r"\brequests\b", r"\burllib\b", r"http\.client", r"\bhttpx\b", r"\baiohttp\b", r"AF_INET", r"AF_INET6",
               r"\bftplib\b", r"\bsmtplib\b", r"huggingface_hub", r"snapshot_download", r"from_pretrained\([^)]*\)"]


def net_hits(root: Path):
    hits = []
    for p in sorted(root.rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for n in NET_NEEDLES[:-1]:
                if re.search(n, line):
                    hits.append((p.name, i, n))
    return hits


def test_no_outbound_network_calls_in_app():
    assert net_hits(APP) == []
    # 対照: unix socket は使っている（検索が動いていること）。
    assert any("AF_UNIX" in p.read_text(encoding="utf-8") for p in APP.rglob("*.py"))


def test_net_scan_detects_planted_calls(tmp_path):
    (tmp_path / "bad.py").write_text("import requests\nimport urllib.request\ns = socket.AF_INET\n")
    assert {h[2] for h in net_hits(tmp_path)} == {r"\brequests\b", r"\burllib\b", "AF_INET"}


def test_model_loading_is_offline_only():
    text = (APP / "mc2ocr" / "backend.py").read_text(encoding="utf-8")
    assert text.count("local_files_only=True") >= 2 and "HF_HUB_OFFLINE" in text


# ---- 見本の答えが無い ----
def norm(s: str) -> str:
    return re.sub(r"[-.·_]", "", re.sub(r"\s+", "", s.upper()))


def full_answers():
    """答え全体を正規化した形。STOP と 35 は短すぎて変数名や数字に当たるので別に扱う。"""
    return sorted({norm(s["expected"]) for s in SAMPLES} - {"STOP", "35"})


def scan_whole_files(root: Path, needles):
    hits = []
    for p in sorted(root.rglob("*.py")) + sorted(root.rglob("*.txt")) + sorted(root.rglob("*.md")):
        if "__pycache__" in p.parts:
            continue
        text = norm(p.read_text(encoding="utf-8", errors="replace"))
        hits += [(str(p.relative_to(root)), n) for n in needles if n in text]
    return hits


def test_sample_table_is_what_we_think():
    assert [s["n"] for s in SAMPLES] == list(range(1, 11))
    assert full_answers() == sorted(["7ABC123", "京A12345", "JHT2951", "5XYZ891", "沪B88888", "SPEEDLIMIT65", "ROADWORKAHEAD"])


def test_app_has_no_sample_answers_in_any_file():
    assert scan_whole_files(APP, full_answers()) == []


def test_sample_digit_tokens_not_in_app():
    # 見本の答えを空白で割った語のうち、数字を含み 3 字以上の物（2951・12345・88888・891 を含む語など）
    toks = sorted({norm(t) for s in SAMPLES for t in re.split(r"\s+", s["expected"]) if re.search(r"\d", t) and len(norm(t)) >= 3})
    assert toks and scan_whole_files(APP, toks) == []


def prompt_and_tables_text() -> str:
    """指示の文と後処理の表（STOP と 35 は、この 2 つの中だけを見る）。"""
    from mc2ocr import postprocess, prompt
    parts = [prompt.PROMPT, " ".join(postprocess.US_STATES), postprocess.PREFIX_RE.pattern]
    return "\n".join(parts)


def test_prompt_and_tables_have_no_stop_or_35():
    text = prompt_and_tables_text()
    assert not re.search(r"\bSTOP\b", text, re.I)
    assert not re.search(r"(?<!\d)35(?!\d)", text)


def test_prompt_examples_use_invented_values():
    from mc2ocr import prompt
    for n in full_answers():
        assert n not in norm(prompt.PROMPT)


def test_scan_detects_a_planted_answer(tmp_path):
    needles = full_answers()
    (tmp_path / "planted.py").write_text(f"PROMPT = 'for example {SAMPLES[0]['expected'].lower()}'\n", encoding="utf-8")
    (tmp_path / "planted2.py").write_text("X = '京 A-12345'\nY = 'road work ahead'\n", encoding="utf-8")
    hits = scan_whole_files(tmp_path, needles)
    assert ("planted.py", "7ABC123") in hits and ("planted2.py", "京A12345") in hits and ("planted2.py", "ROADWORKAHEAD") in hits


def test_stop_scan_detects_planted_prompt():
    assert re.search(r"\bSTOP\b", "a stop sign", re.I) and re.search(r"(?<!\d)35(?!\d)", "limit 35 mph")
