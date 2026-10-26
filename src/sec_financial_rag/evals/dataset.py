"""Build the golden eval set (spec §5): FinanceBench slice + custom questions.

Run: uv run --env-file .env python -m sec_financial_rag.evals.dataset
The output is generated, not committed (FinanceBench's text has no license);
custom questions are ours and are committed.
"""

import json
import os
from pathlib import Path

from sec_financial_rag.evals.financebench import (
    MAPPING_PATH,
    load_financebench,
    to_eval_record,
)
from sec_financial_rag.ingestion.edgar_downloader import make_client

CUSTOM_PATH = Path("data/golden_eval/custom_questions.jsonl")  # committed
DATASET_PATH = Path("data/golden_eval/eval_dataset.jsonl")  # generated, git-ignored


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_records(*groups: list[dict]) -> list[dict]:
    """Concatenate record groups in order; every id must be unique."""
    merged, seen = [], set()
    for record in (r for group in groups for r in group):
        if record["id"] in seen:
            raise ValueError(f"duplicate eval id: {record['id']}")
        seen.add(record["id"])
        merged.append(record)
    return merged


def financebench_records(questions: list[dict], mapping: dict[str, dict]) -> list[dict]:
    """Questions whose document resolved to a verified filing."""
    return [
        to_eval_record(q, mapping[q["doc_name"]])
        for q in questions
        if mapping.get(q["doc_name"], {}).get("status", "").startswith("verified")
    ]


def main() -> None:
    with make_client(os.environ["SEC_USER_AGENT"]) as client:
        questions, _ = load_financebench(client)  # cached after the first download
    mapping = json.loads(MAPPING_PATH.read_text())
    fb = financebench_records(questions, mapping)
    custom = read_jsonl(CUSTOM_PATH) if CUSTOM_PATH.exists() else []
    records = merge_records(fb, custom)
    DATASET_PATH.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(
        f"{len(fb)} FinanceBench + {len(custom)} custom = {len(records)} → {DATASET_PATH}"
    )


if __name__ == "__main__":
    main()
