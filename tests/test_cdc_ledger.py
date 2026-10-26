import uuid

import psycopg
import pytest

from sec_financial_rag.ingestion.cdc_ledger import (
    apply_schema,
    classify_section,
    content_sha256,
    record_filing,
    record_section,
)

EXP1, EXP2 = "exp1_fixed512_bge_m3", "exp2_semantic_bge_m3"

APPLE_10K = {
    "form": "10-K",
    "accession_number": "0000320193-25-000079",
    "report_date": "2025-09-27",
    "primary_document": "aapl-20250927.htm",
}


def test_content_sha256_ignores_whitespace_only_changes():
    assert content_sha256("Revenue  rose\n5%") == content_sha256(" Revenue rose 5% ")


def test_content_sha256_detects_real_changes():
    assert content_sha256("Revenue rose 5%") != content_sha256("Revenue rose 6%")
    # Regression guard for the tokens() trap: parentheses mean a negative number.
    assert content_sha256("Net income (1,234)") != content_sha256("Net income 1,234")


def test_content_sha256_is_64_hex_chars():
    digest = content_sha256("x")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_apply_schema_is_idempotent(db):
    apply_schema(db)
    apply_schema(db)  # second run must not fail
    tables = {
        row[0]
        for row in db.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        )
    }
    assert {"filings", "ingestion_ledger"} <= tables


def test_record_filing_reports_new_unchanged_changed(db):
    apply_schema(db)
    assert record_filing(db, "AAPL", 320193, APPLE_10K, "aaa") == "new"
    assert record_filing(db, "AAPL", 320193, APPLE_10K, "aaa") == "unchanged"
    assert record_filing(db, "AAPL", 320193, APPLE_10K, "bbb") == "changed"
    stored = db.execute(
        "SELECT file_sha256, count(*) OVER () FROM filings WHERE accession_number = %s",
        (APPLE_10K["accession_number"],),
    ).fetchone()
    assert stored == ("bbb", 1)  # hash updated in place, no duplicate row


def test_classify_section_compares_with_current_row_only(db):
    apply_schema(db)
    record_filing(db, "AAPL", 320193, APPLE_10K, "file-hash")
    acc = APPLE_10K["accession_number"]
    assert classify_section(db, EXP1, acc, "FULL", "aaa") == "new"

    db.execute(
        "INSERT INTO ingestion_ledger (collection, accession_number, section_id, content_sha256, is_superseded)"
        " VALUES (%s, %s, 'FULL', 'old', true), (%s, %s, 'FULL', 'aaa', false)",
        (EXP1, acc, EXP1, acc),
    )
    assert classify_section(db, EXP1, acc, "FULL", "aaa") == "unchanged"
    assert classify_section(db, EXP1, acc, "FULL", "bbb") == "changed"
    # A superseded version must not count as "unchanged".
    assert classify_section(db, EXP1, acc, "FULL", "old") == "changed"


def current_rows(db, acc):
    return db.execute(
        "SELECT content_sha256, vector_point_ids FROM ingestion_ledger"
        " WHERE accession_number = %s AND section_id = 'FULL' AND NOT is_superseded",
        (acc,),
    ).fetchall()


def test_record_section_supersedes_and_returns_old_point_ids(db):
    apply_schema(db)
    record_filing(db, "AAPL", 320193, APPLE_10K, "file-hash")
    acc = APPLE_10K["accession_number"]
    old_ids = [uuid.uuid4(), uuid.uuid4()]
    new_ids = [uuid.uuid4()]

    assert record_section(db, EXP1, acc, "FULL", "v1", old_ids) == []  # new section
    assert record_section(db, EXP1, acc, "FULL", "v2", new_ids) == old_ids  # changed
    assert current_rows(db, acc) == [("v2", new_ids)]  # exactly one current row


def test_record_section_is_all_or_nothing(db):
    apply_schema(db)
    record_filing(db, "AAPL", 320193, APPLE_10K, "file-hash")
    acc = APPLE_10K["accession_number"]
    ids = [uuid.uuid4()]
    record_section(db, EXP1, acc, "FULL", "v1", ids)

    with pytest.raises(psycopg.Error):  # the INSERT fails: not a UUID
        record_section(db, EXP1, acc, "FULL", "v2", ["not-a-uuid"])
    assert current_rows(db, acc) == [("v1", ids)]  # the supersede was undone too


def test_each_collection_tracks_its_own_current_version(db):
    # Learning record 0016: Exp-2 must not see Exp-1's work as "unchanged".
    apply_schema(db)
    record_filing(db, "AAPL", 320193, APPLE_10K, "file-hash")
    acc = APPLE_10K["accession_number"]
    record_section(db, EXP1, acc, "FULL", "same-text", [uuid.uuid4()])

    assert classify_section(db, EXP2, acc, "FULL", "same-text") == "new"
    record_section(db, EXP2, acc, "FULL", "same-text", [uuid.uuid4()])
    current = db.execute(
        "SELECT count(*) FROM ingestion_ledger"
        " WHERE accession_number = %s AND NOT is_superseded",
        (acc,),
    ).fetchone()[0]
    assert current == 2  # one current row per collection
