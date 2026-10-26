"""主催の見本の束での試験。環境変数 X052_KIT（starter-kit のフォルダ）があるときだけ走る。"""
import json
import os
import shutil
from pathlib import Path

import pytest

from x052 import indexer
from x052.answerer import Answerer
from x052.backend import StubBackend

KIT = os.environ.get("X052_KIT")
pytestmark = pytest.mark.skipif(not KIT, reason="X052_KIT not set")


@pytest.fixture
def kit_corpus(tmp_path):
    dst = tmp_path / "mc3-corpus"
    shutil.copytree(Path(KIT) / "mc3-corpus", dst)
    (dst / "archive").mkdir(exist_ok=True)
    audit = dst / "vendor" / "internal_audit.txt"
    audit.chmod(0o000)  # tmp の写しの中だけ
    try:
        yield dst
    finally:
        audit.chmod(0o644)


def test_kit_walk_13_files(kit_corpus):
    chunks, manifest = indexer.walk_corpus(str(kit_corpus))
    files = [m for m in manifest if m["kind"] != "dir"]
    assert len(files) == 13
    m = {x["path"]: x for x in manifest}
    assert "unreadable" in m["vendor/internal_audit.txt"]["reason"]
    assert m["vendor/supplier_agreement_ENCRYPTED.pdf"]["reason"] == "encrypted"
    assert m["vendor/telemetry_capture.dat"]["reason"] == "unknown binary type"
    assert m["archive"]["reason"] == "empty directory"
    assert sum(1 for x in files if x["status"] == "image") == 2
    assert sum(1 for x in files if x["status"] == "indexed") == 8
    assert sum(1 for x in files if x["status"] == "skipped") == 3


def test_kit_q9_two_hop_includes_log_and_csv(kit_corpus):
    q9 = [q for q in json.load(open(Path(KIT) / "sample-questions.json"))["queries"] if q["n"] == 9][0]
    chunks, _ = indexer.walk_corpus(str(kit_corpus))
    a = Answerer(chunks, StubBackend())
    paths = {chunks[i]["path"] for i in a.select(q9["query"])}
    assert "logs/prod_inference_2026-09-02.log" in paths
    assert "support/bug_database.csv" in paths


def _q9():
    return [q for q in json.load(open(Path(KIT) / "sample-questions.json"))["queries"] if q["n"] == 9][0]["query"]


def test_kit_q9_chunk_level_row_present(kit_corpus):
    """ORR-1847 の行（fixed_in=4.3.2）の塊そのものが入っている。"""
    chunks, _ = indexer.walk_corpus(str(kit_corpus))
    a = Answerer(chunks, StubBackend())
    sel = a.select(_q9())
    rows = [i for i in sel if "ORR-1847" in chunks[i]["text"] and "4.3.2" in chunks[i]["text"]
            and chunks[i]["path"] == "support/bug_database.csv"]
    assert rows


def test_kit_q9_second_hop_alone_reaches_csv_row(kit_corpus, monkeypatch):
    """1 段目を 1 塊に絞り、識別子の 2 段目だけで不具合の表の行に届くか。"""
    from x052 import answerer
    chunks, _ = indexer.walk_corpus(str(kit_corpus))
    q = _q9()
    a = Answerer(chunks, StubBackend())
    log_chunk = [i for i, c in enumerate(chunks) if c["path"].startswith("logs/")]
    from x052.retrieve import second_hop
    adds = second_hop(chunks, a.id_table, log_chunk, q)
    assert any(ident == "ORR-1847" and chunks[j]["path"] == "support/bug_database.csv"
               and "4.3.2" in chunks[j]["text"] for j, ident, _ in adds)
    assert len(adds) <= 4
    monkeypatch.setattr(answerer, "RETRIEVE_K", 1)
    monkeypatch.setattr(answerer, "TOP_K", 1)
    sel = a.select(q)
    assert len(sel) <= 1 + 4


def test_kit_q9_hop_follows_orr1847_first(kit_corpus):
    q = _q9()
    chunks, _ = indexer.walk_corpus(str(kit_corpus))
    a = Answerer(chunks, StubBackend())
    a.select(q)
    idents = [ident for ident, _ in a.links.values()]
    assert "ORR-1847" in idents                      # 不具合の番号をたどる
    assert len(a.links) <= 4
    per_file = {}
    for j in a.links:
        per_file[chunks[j]["path"]] = per_file.get(chunks[j]["path"], 0) + 1
    assert max(per_file.values()) <= 2               # ファイルごとに 2 塊まで
    # 出るファイルの少ない ORR-1847（2 ファイル）が、製品名 TQ-40 より先に選ばれる
    if "TQ-40" in idents:
        assert idents.index("ORR-1847") < idents.index("TQ-40")


def test_kit_q2_no_withdrawn_after_second_hop(kit_corpus):
    q2 = [q for q in json.load(open(Path(KIT) / "sample-questions.json"))["queries"] if q["n"] == 2][0]["query"]
    chunks, _ = indexer.walk_corpus(str(kit_corpus))
    a = Answerer(chunks, StubBackend())
    paths = {chunks[i]["path"] for i in a.select(q2)}
    assert "specs/tq40_datasheet_r1_WITHDRAWN.pdf" not in paths
    assert "planning/roadmap_fy27.docx" in paths
