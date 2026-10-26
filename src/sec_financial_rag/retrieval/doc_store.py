"""Parent-document store for Exp-3 (spec §4.3 Strategy 3), in the ledger's Postgres.

Same database as the CDC ledger, so parents and ledger rows commit in one
transaction: a crash can't leave a ledger row without its parents.

Order when a section changes (mirrors the Qdrant rule in indexer_exp2):
put_parents + record_section -> commit -> delete old children -> delete_stale_parents.
Old parents outlive old children, so a search never finds a child without a parent.
"""

import uuid
from collections.abc import Callable
from uuid import UUID

import psycopg

from sec_financial_rag.retrieval.qdrant_store import NAMESPACE


def parent_id(
    collection: str, accession: str, section_id: str, index: int, sha: str
) -> UUID:
    """Deterministic, like point_id: a retry hits the same row; a new sha, new ids."""
    return uuid.uuid5(
        NAMESPACE, f"parent|{collection}|{accession}|{section_id}|{index}|{sha}"
    )


def put_parents(
    conn: psycopg.Connection,
    collection: str,
    accession: str,
    section_id: str,
    sha: str,
    texts: list[str],
) -> list[UUID]:
    """Store one section's parents in order; returns their ids. Does NOT commit."""
    ids = [
        parent_id(collection, accession, section_id, i, sha) for i in range(len(texts))
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO parent_documents (parent_id, collection, accession_number,"
            " section_id, content_sha256, parent_index, text)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (parent_id) DO NOTHING",  # a retry after a crash is harmless
            [
                (pid, collection, accession, section_id, sha, i, text)
                for i, (pid, text) in enumerate(zip(ids, texts))
            ],
        )
    return ids


def get_parents(conn: psycopg.Connection, ids: list[UUID]) -> dict[UUID, str]:
    """Parent texts by id, in one query; missing ids are simply absent."""
    rows = conn.execute(
        "SELECT parent_id, text FROM parent_documents WHERE parent_id = ANY(%s)",
        (list(ids),),
    ).fetchall()
    return dict(rows)


def delete_stale_parents(
    conn: psycopg.Connection,
    collection: str,
    accession: str,
    section_id: str,
    current_sha: str,
) -> int:
    """Delete a section's parents from older versions; returns how many. Does NOT commit."""
    return conn.execute(
        "DELETE FROM parent_documents WHERE collection = %s AND accession_number = %s"
        " AND section_id = %s AND content_sha256 <> %s",
        (collection, accession, section_id, current_sha),
    ).rowcount


def expand_to_parents(
    children: list[dict], fetch: Callable[[list[UUID]], dict[UUID, str]]
) -> list[dict]:
    """Ranked child hits -> ranked parents (spec §4.3 retrieval mechanics).

    A parent ranks where its best child ranked and takes that child's payload and
    score; `text` becomes the parent's text and the child's moves to `child_text`.
    `fetch` is get_parents bound to a connection (tests pass a dict lookup). A
    missing parent raises KeyError: the deletion order means it should never happen.
    """
    best: dict[str, dict] = {}
    for child in children:
        best.setdefault(child["parent_id"], child)  # first = highest-ranked child
    texts = fetch([UUID(pid) for pid in best])
    return [
        {**child, "text": texts[UUID(pid)], "child_text": child["text"]}
        for pid, child in best.items()
    ]
