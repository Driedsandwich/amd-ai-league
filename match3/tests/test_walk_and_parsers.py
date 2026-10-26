import os

from x052 import indexer, parsers
from x052.retrieve import BM25


def by_path(manifest):
    return {m["path"]: m for m in manifest}


def test_walk_survives_everything(corpus, unreadable):
    chunks, manifest = indexer.walk_corpus(str(corpus))
    m = by_path(manifest)
    assert m["vendor/audit.txt"]["status"] == "skipped"
    assert "unreadable" in m["vendor/audit.txt"]["reason"]
    assert m["vendor/supplier_ENCRYPTED.pdf"]["reason"] == "encrypted"
    assert m["vendor/capture.dat"]["reason"] == "unknown binary type"
    assert m["vendor/empty.txt"]["reason"].startswith("empty")
    assert m["archive"]["reason"] == "empty directory"
    assert m["support/label.jpg"]["status"] == "image"
    # 読めた物は全部入っている（読めない物の後ろにあるファイルも）
    indexed = {p for p, v in m.items() if v["status"] == "indexed"}
    for p in ["engineering/svc.py", "logs/run.log", "planning/roadmap_fy27.docx",
              "specs/widget_datasheet_r2.pdf", "support/bugs.csv", "support/rma_parts.xlsx"]:
        assert p in indexed, p
    # 暗号化 PDF の中身（6412）は索引に出ない
    assert not any("6412" in c["text"] for c in chunks)
    # 全ファイルが目録に出る（落ちたファイルが無い）
    on_disk = set()
    for d, ds, fs in os.walk(corpus):
        on_disk.update(os.path.relpath(os.path.join(d, f), corpus) for f in fs)
    assert on_disk <= set(m)


def test_walk_order_is_sorted(corpus):
    _, manifest = indexer.walk_corpus(str(corpus))
    files = [m["path"] for m in manifest if m["kind"] != "dir"]
    dirs_files = sorted(files, key=lambda p: (os.path.dirname(p), os.path.basename(p)))
    assert files == dirs_files


def test_walk_nonexistent_root_does_not_raise(tmp_path):
    chunks, manifest = indexer.walk_corpus(str(tmp_path / "nope"))
    assert chunks == []


def test_parsers_return_text(corpus):
    def txt(rel):
        return "\n".join(t for _, t in parsers.parse_file(str(corpus / rel)))

    assert "94 C" in txt("specs/widget_datasheet_r2.pdf")
    t = txt("planning/roadmap_fy27.docx")
    assert "Quarter=Q3 FY27" in t and "Product=W60" in t
    t = txt("support/rma_parts.xlsx")
    assert "assembly=W9 field-replaceable fan" in t and "number=ZZ-FAN-1-B" in t
    t = txt("support/bugs.csv")
    assert "ticket=ZZ-100" in t and "fixed_in=4.3.2" in t
    assert "E7731" in txt("logs/run.log")
    assert "DEFAULT_BATCH_TIMEOUT_S = 180" in txt("engineering/svc.py")


def test_parser_error_types(corpus):
    import pytest
    with pytest.raises(parsers.EncryptedError):
        parsers.parse_file(str(corpus / "vendor/supplier_ENCRYPTED.pdf"))
    with pytest.raises(parsers.UnknownTypeError):
        parsers.parse_file(str(corpus / "vendor/capture.dat"))
    with pytest.raises(parsers.EmptyFileError):
        parsers.parse_file(str(corpus / "vendor/empty.txt"))
    with pytest.raises(parsers.ImageFile):
        parsers.parse_file(str(corpus / "support/label.jpg"))


def test_bm25_finds_the_right_file(corpus):
    chunks, _ = indexer.walk_corpus(str(corpus))
    bm = BM25(chunks)
    top = [chunks[i]["path"] for i in bm.search("What error code is logged when the thermal throttle engages?", 8)]
    assert top[0] == "logs/run.log"
    top = [chunks[i]["path"] for i in bm.search("Which firmware version fixed ticket ZZ-100?", 8)]
    assert "support/bugs.csv" in top


def test_two_hop_id_table(corpus):
    from x052.retrieve import build_id_table, second_hop
    chunks, _ = indexer.walk_corpus(str(corpus))
    table = build_id_table(chunks)
    assert {chunks[j]["path"] for j in table["ZZ-100"]} == {"logs/run.log", "support/bugs.csv"}
    bm = BM25(chunks)
    q = "The log shows a thermal throttle. Which release fixed it?"
    first = bm.search(q, 1)
    adds = second_hop(chunks, table, first, q)
    paths = {chunks[i]["path"] for i in first} | {chunks[j]["path"] for j, _, _ in adds}
    assert "logs/run.log" in paths and "support/bugs.csv" in paths
    assert len(adds) <= 4
    # 問いにある識別子では辿らない
    assert second_hop(chunks, table, first, q + " ZZ-100") == []
