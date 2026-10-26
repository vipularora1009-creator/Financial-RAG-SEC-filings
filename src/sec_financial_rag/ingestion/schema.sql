-- CDC ledger schema (spec §4.2). Idempotent: safe to apply on every run.

-- One row per filing we know about (acquisition stage).
CREATE TABLE IF NOT EXISTS filings (
    accession_number  TEXT PRIMARY KEY,          -- SEC's unique filing ID (natural key)
    ticker            TEXT NOT NULL,
    cik               INTEGER NOT NULL,
    form_type         TEXT NOT NULL,             -- 10-K, 10-Q, 10-K/A, 10-Q/A
    report_date       DATE NOT NULL,             -- period end
    primary_document  TEXT NOT NULL,
    file_sha256       TEXT,                      -- hash of the downloaded bytes
    first_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per section version (processing stage).
CREATE TABLE IF NOT EXISTS ingestion_ledger (
    id                BIGSERIAL PRIMARY KEY,
    collection        TEXT NOT NULL,             -- Qdrant collection = experiment config, e.g. exp1_fixed512_bge_m3
    accession_number  TEXT NOT NULL REFERENCES filings (accession_number),
    section_id        TEXT NOT NULL,             -- Week 1: 'FULL'; Week 2: Item_7, Item_8_Note_4, ...
    content_sha256    TEXT NOT NULL,             -- hash of the normalized section text
    vector_point_ids  UUID[] NOT NULL DEFAULT '{}',
    is_superseded     BOOLEAN NOT NULL DEFAULT false,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Migration 2026-10-24: ledger rows are per collection, so Exp-2 doesn't see Exp-1's work
-- as "unchanged" (learning record 0016). ADD COLUMN ... NOT NULL without a default only
-- works because the table was empty when this ran; with rows it would need a DEFAULT.
ALTER TABLE ingestion_ledger ADD COLUMN IF NOT EXISTS collection TEXT NOT NULL;
DROP INDEX IF EXISTS one_current_section;  -- old index, keyed without collection

-- At most one *current* version per (collection, filing, section); superseded versions are kept.
CREATE UNIQUE INDEX IF NOT EXISTS one_current_section_per_collection
    ON ingestion_ledger (collection, accession_number, section_id)
    WHERE NOT is_superseded;

-- Cache of EDGAR's older submission files (filings.files), so re-runs don't refetch them.
-- A hit requires the same name, date range AND count as the fresh index (verify, don't assume).
CREATE TABLE IF NOT EXISTS submission_files (
    name          TEXT PRIMARY KEY,          -- e.g. CIK0000019617-submissions-001.json
    filing_from   DATE NOT NULL,
    filing_to     DATE NOT NULL,
    filing_count  INTEGER NOT NULL,
    rows          JSONB NOT NULL,            -- our filtered rows: list of dicts
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Exp-3 document store (spec §4.3 Strategy 3): parent nodes the LLM reads.
-- Children live in Qdrant with a parent_id payload; a parent's id includes its
-- section's sha, so a new section version never overwrites a row still in use.
CREATE TABLE IF NOT EXISTS parent_documents (
    parent_id         UUID PRIMARY KEY,
    collection        TEXT NOT NULL,
    accession_number  TEXT NOT NULL REFERENCES filings (accession_number),
    section_id        TEXT NOT NULL,
    content_sha256    TEXT NOT NULL,             -- section version the parent came from
    parent_index      INTEGER NOT NULL,          -- order within the section
    text              TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS parent_documents_by_section
    ON parent_documents (collection, accession_number, section_id);
