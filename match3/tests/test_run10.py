import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import run10  # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
APP = Path(__file__).resolve().parent.parent / "app" / "app.py"


def test_answer_matching_rules():
    m = run10.answer_matches
    assert m("94 C", "94", ["94°C"]) and m("94°C", "94", ["94 C"]) and m(" 94 ", "94", [])
    assert m("q3  fy27", "Q3 FY27", ["Q3FY27"]) and m("", "", [])
    assert not m("95", "94", ["94 C"]) and not m("x", "", [])


def test_parse_vram():
    assert run10.parse_vram_used('{"card0": {"VRAM Total Memory (B)": "100", "VRAM Total Used Memory (B)": "2147483648"}}') == 2 * 1024 ** 3
    assert run10.parse_vram_used('[{"mem_usage": {"used_vram": {"value": 1024, "unit": "MB"}}}]') == 1024 ** 3
    assert run10.parse_vram_used("not json") is None


def test_run10_with_stub(corpus, tmp_path):
    qs = {"queries": [
        {"n": 1, "query": "What is the maximum junction temperature of the Widget W9?", "expected_answer": "94",
         "answer_aliases": ["94 C"], "expected_citations": ["specs/widget_datasheet_r2.pdf"]},
        {"n": 2, "query": "unrelated", "expected_answer": "", "expected_citations": []},
    ]}
    qf = tmp_path / "q.json"
    qf.write_text(json.dumps(qs))
    rep = tmp_path / "rep.json"
    r = subprocess.run([sys.executable, str(TOOLS / "run10.py"), "--app", str(APP), "--corpus", str(corpus),
                        "--queries", str(qf), "--index-dir", str(tmp_path / "idx"), "--output-dir",
                        str(tmp_path / "out"), "--python", sys.executable, "--env", "X052_BACKEND=stub",
                        "--env", "X052_STUB_ANSWER=94", "--env", "X052_READY_TIMEOUT=60", "--report", str(rep)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(rep.read_text())
    q1 = rep["queries"][0]
    assert q1["valid"] and q1["answer_ok"] and q1["citations_ok"] and q1["server_alive_before"] is True
    assert rep["gates"]["outputs_valid"] and rep["gates"]["index_time"] and rep["gates"]["each_query_time"]
    assert rep["gates"]["vram_1_to_48gib"] is None  # この Mac には amd-smi が無い
    assert not (tmp_path / "idx" / "server.pid").exists() or not run10.server_alive(tmp_path / "idx")
