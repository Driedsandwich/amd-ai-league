import json

from x052 import answerer, backend, indexer, versions
from x052.answerer import Answerer, contains_value, normalize_answer, parse_model_json
from x052.retrieve import build_id_table, second_hop


class Fake:
    """決まった JSON を返す疑似の模型。sources は渡された塊の全部のファイル。"""

    def __init__(self, answer, answerable=True, sources=None):
        self.answer, self.answerable, self.sources = answer, answerable, sources

    def generate(self, prompt, n, max_time=None):
        import re
        paths = list(dict.fromkeys(re.findall(r"^### FILE: (.+?) \[", prompt, flags=re.M)))
        return json.dumps({"answerable": self.answerable, "answer": self.answer,
                           "sources": self.sources if self.sources is not None else paths})


def mk(path, text, loc="l"):
    return {"path": path, "loc": loc, "text": text}


def numbered(chunks):
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


# ---- 版（指摘 7）----
def test_version_key_and_withdrawn_marks(corpus):
    assert versions.base_key("specs/widget_datasheet_r1_WITHDRAWN.pdf") == versions.base_key(
        "specs/widget_datasheet_r2.pdf")
    chunks, _ = indexer.walk_corpus(str(corpus))
    t = versions.VersionTable(chunks)
    assert t.current("specs/widget_datasheet_r1_WITHDRAWN.pdf") == "specs/widget_datasheet_r2.pdf"
    order = list(range(len(chunks)))
    new, replaced = versions.rerank(chunks, order, t)
    paths = {chunks[i]["path"] for i in new}
    assert "specs/widget_datasheet_r1_WITHDRAWN.pdf" not in paths and "specs/widget_datasheet_r2.pdf" in paths
    assert replaced == {"specs/widget_datasheet_r1_WITHDRAWN.pdf": "specs/widget_datasheet_r2.pdf"}


def test_version_counterexample_new_file_says_old_is_obsolete():
    """無印の r1 と、本文に「r1 is obsolete」と書く r2: r2 が残る（旧実装は r2 を消した）。"""
    chunks = numbered([mk("doc_r1.txt", "Widget limit is 105 C."),
                       mk("doc_r2.txt", "Widget limit is 94 C. Note: r1 is obsolete.")])
    t = versions.VersionTable(chunks)
    assert t.current("doc_r1.txt") == "doc_r2.txt" and not t.is_old("doc_r2.txt")
    a = Answerer(chunks, Fake("94"))
    idx = a.select("What is the Widget limit?")
    assert {chunks[i]["path"] for i in idx} == {"doc_r2.txt"}
    assert a.answer("What is the Widget limit?") == ("94", ["doc_r2.txt"], 0.9)
    # 旧版にしか無い値は答えない（根拠が旧版だけ）
    assert Answerer(chunks, Fake("105")).answer("What is the Widget limit?") == ("", [], 0.0)


def test_head_statements():
    hs = versions.head_statements
    assert hs("This document is WITHDRAWN.", 1) == (True, set(), False)
    assert hs("Note: r1 is obsolete.", 2) == (False, {1}, False)
    assert hs("Superseded by r2.", 1) == (True, set(), False)
    assert hs("This revision supersedes the previous revision.", 2) == (False, set(), True)
    assert hs("Normal text only.", 1) == (False, set(), False)
    # 先頭 500 字より後ろの言明は見ない
    assert hs("x " * 400 + "WITHDRAWN", 1) == (False, set(), False)


def test_old_version_retrieved_pulls_in_current():
    chunks = numbered([mk("sheet_r1.txt", "zebra gadget limit 105 C"), mk("sheet_r2.txt", "gadget limit 94 C"),
                       mk("sheet_r2.txt", "Note r1 is obsolete", "l2")])
    a = Answerer(chunks, Fake("94"))
    idx = a.select("zebra gadget limit")
    paths = {chunks[i]["path"] for i in idx}
    assert "sheet_r1.txt" not in paths and "sheet_r2.txt" in paths


# ---- JSON の型（指摘 5）----
def test_parse_model_json_type_checks():
    ok = '{"answerable": true, "answer": "1", "sources": ["a"]}'
    assert parse_model_json("```json\n" + ok + "\n```")["answer"] == "1"
    assert parse_model_json('{"answerable": "false", "answer": "1", "sources": []}') is None
    assert parse_model_json('{"answer": "1", "sources": []}') is None  # answerable 欠落
    assert parse_model_json('{"answerable": true, "answer": ["1"], "sources": []}') is None
    assert parse_model_json('{"answerable": true, "answer": "1", "sources": "a"}') is None
    assert parse_model_json('{"answerable": true, "answer": "1", "sources": [1]}') is None
    assert parse_model_json("no json here") is None
    two = '{"x": 1} then ' + ok
    assert parse_model_json(two)["answer"] == "1"  # 型を満たす物だけ採る


def test_bad_model_output_gives_empty():
    chunks = numbered([mk("a.txt", "value 94")])

    class Bad:
        def generate(self, prompt, n, max_time=None):
            return '{"answerable": "yes", "answer": "94", "sources": ["a.txt"]}'

    assert Answerer(chunks, Bad()).answer("value?") == ("", [], 0.0)


# ---- 答えの整え・持ち主（指摘 6・9）----
def test_normalize_answer():
    assert normalize_answer("  94. ") == "94"
    assert normalize_answer('"Q3 FY27"。') == "Q3 FY27"


def test_contains_value_boundaries():
    assert contains_value("tjmax 94 C", "94") and contains_value("tjmax=94°C", "94 C")
    assert not contains_value("batch 1947", "94") and not contains_value("rev 94.5", "94")
    assert contains_value("fixed_in=4.3.2", "4.3.2") and contains_value("Quarter=Q3 FY27", "Q3FY27")


def test_long_answers_reduce_to_typed_value_present_in_owner():
    chunks = numbered([mk("ds.txt", "Orrery TQ-40 datasheet. Maximum junction temperature 94 C."),
                       mk("notes.txt", "Firmware fixed in 4.3.2 for ORR-1847.")])
    a = Answerer(chunks, Fake("x"))
    idx = [0, 1]
    q = "What is the maximum junction temperature of the TQ-40?"
    assert a.resolve_value("The TQ-40 maximum junction temperature is 94 C.", q, idx)[0] == "94"
    q = "Which firmware version fixed the ticket?"
    assert a.resolve_value("Firmware version is 4.3.2.", q, idx)[0] == "4.3.2"  # 4 語でも文を値にする
    assert a.resolve_value("The fix shipped in version 9.9.9 of the firmware package ok", q, idx) is None
    assert answerer.value_pattern("In which quarter does it ship?").fullmatch("Q3 FY27")
    assert answerer.value_pattern("Which backplane pin carries X?").fullmatch("B14")
    assert answerer.value_pattern("What board revision is printed?").fullmatch("REV-C2")


def test_answer_without_owner_is_refused():
    chunks = numbered([mk("a.txt", "limit 94 C")])
    assert Answerer(chunks, Fake("999")).answer("What is the limit?") == ("", [], 0.0)
    assert Answerer([], Fake("1")).answer("anything") == ("", [], 0.0)


def test_owner_is_one_file_and_prefers_model_sources_then_order():
    chunks = numbered([mk("a.txt", "limit 94 C here"), mk("b.txt", "limit 94 C there")])
    a = Answerer(chunks, Fake("94", sources=["b.txt"]))
    assert a.answer("What is the limit?")[1] == ["b.txt"]
    a = Answerer(chunks, Fake("94"))  # sources=全部でも持ち主は 1 つ
    assert len(a.answer("What is the limit?")[1]) == 1


def test_origin_added_only_when_owner_reached_through_identifier():
    log = mk("log.txt", "2026 ERROR E7731 thermal throttle engaged. incident logged against ZZ-100 production log")
    row = mk("bugs.csv", "ticket=ZZ-100 | summary=sensor | fixed_in=4.3.2", "row 2")
    filler = [mk(f"f{i}.txt", f"filler text {i}") for i in range(3)]
    chunks = numbered([log, row] + filler)
    # 問 9 型: ログ（問いに近い）から識別子で表の行へ。答えの持ち主は表・起点のログも引用
    q9 = "The production log shows a thermal throttle incident. Which firmware release fixed the defect?"
    ans, cites, _ = Answerer(chunks, Fake("4.3.2")).answer(q9)
    assert ans == "4.3.2" and set(cites) == {"bugs.csv", "log.txt"}
    # 問 5 型: 答えはログの中だけ。表は同じ識別子を持つが引かない
    q5 = "What error code is logged when the thermal throttle engages?"
    ans, cites, _ = Answerer(chunks, Fake("E7731")).answer(q5)
    assert ans == "E7731" and cites == ["log.txt"]
    # 問 4 型: 問いに識別子がある → 起点を足さない
    q4 = "Which firmware version fixed ticket ZZ-100?"
    ans, cites, _ = Answerer(chunks, Fake("4.3.2")).answer(q4)
    assert cites == ["bugs.csv"]


# ---- 2 段目（指摘 8）----
def test_second_hop_scoring_per_file_cap_and_order():
    chunks = numbered([
        mk("origin.txt", "see AAA-11 and BBB-22 here"),
        mk("x.txt", "AAA-11 one"), mk("x.txt", "AAA-11 two", "l2"), mk("x.txt", "AAA-11 three", "l3"),
        mk("y.txt", "BBB-22 only"), mk("z.txt", "BBB-22 more"), mk("w.txt", "BBB-22 other"),
    ])
    table = build_id_table(chunks)
    scores = {i: 1.0 for i in range(len(chunks))}
    adds = second_hop(chunks, table, [0], "question", scores=scores)
    files = [chunks[j]["path"] for j, _, _ in adds]
    assert len(adds) <= 4 and all(files.count(f) <= 2 for f in set(files))
    # 出るファイルが少ない識別子（AAA-11＝2 ファイル）が先
    assert adds[0][1] == "AAA-11"
    # 問いに識別子があれば対象外
    assert all(ident != "AAA-11" for _, ident, _ in second_hop(chunks, table, [0], "about AAA-11", scores=scores))


def test_second_hop_does_not_cut_common_ids_when_chunk_is_close():
    chunks = numbered([mk("origin.txt", "uses PROD-40")] + [mk(f"f{i}.txt", "PROD-40 text") for i in range(6)])
    table = build_id_table(chunks)
    close = {i: 1.0 for i in range(len(chunks))}
    far = {i: 0.0 for i in range(len(chunks))}
    far[0] = 1.0
    assert second_hop(chunks, table, [0], "q", scores=close)  # 多くのファイルに出ても、近ければたどる
    assert second_hop(chunks, table, [0], "q", scores=far) == []  # 近くなければ点が足りない


def test_stub_answerer_shapes(corpus):
    chunks, _ = indexer.walk_corpus(str(corpus))
    a = Answerer(chunks, backend.StubBackend("94"))
    ans, cites, conf = a.answer("What is the maximum junction temperature of the Widget W9?")
    assert isinstance(ans, str) and ans == "94"
    assert isinstance(cites, list) and all(isinstance(c, str) for c in cites)
    assert cites == ["specs/widget_datasheet_r2.pdf"]
