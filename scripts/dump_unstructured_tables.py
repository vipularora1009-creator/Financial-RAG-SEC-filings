"""Dump Unstructured's tables for the ADR-001 comparison (runs OUTSIDE the project env).

Unstructured is not a project dependency, so this runs in a throwaway venv:
  uv venv /tmp/unstr && uv pip install --python /tmp/unstr/bin/python unstructured \
      "spacy==3.8.7" "numpy<2.3" "preshed==3.0.9" "cymem==2.0.11" "murmurhash==1.0.12"
  SCARF_NO_ANALYTICS=true DO_NOT_TRACK=true UNSTRUCTURED_DISABLE_NOTICE=1 \
      /tmp/unstr/bin/python -I scripts/dump_unstructured_tables.py out.json FILE.htm [...]
(the spaCy pins work around an OverflowError loading en_core_web_sm with newer
preshed/cymem; the env vars turn off Unstructured's usage telemetry).
Output: {path: {"seconds": float, "tables": [{"text": str, "html": str}]}}
"""

import json
import sys
import time

from unstructured.partition.html import partition_html

out: dict = {}
for path in sys.argv[2:]:
    t0 = time.perf_counter()
    elements = partition_html(filename=path)
    out[path] = {
        "seconds": round(time.perf_counter() - t0, 2),
        "tables": [
            {"text": e.text, "html": e.metadata.text_as_html or ""}
            for e in elements
            if type(e).__name__ == "Table"
        ],
    }
    print(f"{path}: {len(out[path]['tables'])} tables, {out[path]['seconds']} s")
with open(sys.argv[1], "w") as f:
    json.dump(out, f)
