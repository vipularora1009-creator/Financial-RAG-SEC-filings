"""Sentence embeddings with a disk cache, for semantic breakpoints (spec §4.3).

Embedding every Risk Factors / MD&A sentence costs about two hours on an M1, so
each vector is stored under sha256(model name + sentence) and computed once.
The model name is part of the key: a different model must never reuse old vectors.
SQLite, not the Postgres ledger: this is a disposable cache; deleting the file is a safe reset.
"""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

CACHE_PATH = Path("data/cache/sentence_vectors.sqlite")


class CachedEmbedder:
    """embed(sentences) -> one unit-length float32 row per sentence, in order."""

    def __init__(self, model, model_name: str, path: Path = CACHE_PATH):
        self.model = model  # anything with SentenceTransformer's encode()
        self.model_name = model_name
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
        )

    def key(self, sentence: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{sentence}".encode()).hexdigest()

    def __call__(self, sentences: list[str]) -> np.ndarray:
        keys = [self.key(s) for s in sentences]
        found = self._lookup(set(keys))
        missing = {
            k: s for k, s in zip(keys, sentences) if k not in found
        }  # dedups too
        if missing:
            vectors = self.model.encode(
                list(missing.values()), normalize_embeddings=True
            )
            rows = [
                (k, v.astype(np.float16).tobytes()) for k, v in zip(missing, vectors)
            ]
            with self.db:  # one transaction: commit on success
                self.db.executemany(
                    "INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows
                )
            found |= {
                k: np.asarray(v, dtype=np.float32) for k, v in zip(missing, vectors)
            }
        return (
            np.stack([found[k] for k in keys]) if keys else np.empty((0, 0), np.float32)
        )

    def _lookup(self, keys: set[str]) -> dict[str, np.ndarray]:
        found = {}
        key_list = list(keys)
        for i in range(0, len(key_list), 500):  # stay under SQLite's parameter limit
            batch = key_list[i : i + 500]
            marks = ",".join("?" * len(batch))
            for key, blob in self.db.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({marks})", batch
            ):
                found[key] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        return found
