"""Re-score faithfulness in saved eval reports after a metric change: no new answers.

Run: uv run python -m sec_financial_rag.evals.rescore <report name> [...]
     e.g. exp2_semantic_bge_m3 exp3_parent_child_bge_m3_parents
Each report row keeps the answer and the exact context it was given (runner.py),
so the metric can be recomputed for free. Rewrites <name>_eval.jsonl and its summary.
"""

import json
import sys

from sec_financial_rag.evals.faithfulness import numeric_faithfulness
from sec_financial_rag.evals.runner import REPORT_DIR, summaries


def rescore(rows: list[dict]) -> list[dict]:
    return [
        row
        | {
            "faithfulness": numeric_faithfulness(
                row["answer"], [{"text": t} for t in row["context"]], row["question"]
            )
        }
        for row in rows
    ]


def main(names: list[str]) -> None:
    for name in names:
        path = REPORT_DIR / f"{name}_eval.jsonl"
        rows = rescore([json.loads(line) for line in path.read_text().splitlines()])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = summaries(rows)
        (REPORT_DIR / f"{name}_eval_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        s = summary["all"]
        print(
            f"{name}: faithfulness {s['faithfulness']:.3f},"
            f" fully grounded {s['fully_faithful']:.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
