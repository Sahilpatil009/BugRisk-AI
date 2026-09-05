"""Generate CodeBERT embeddings for the extracted commit sources.

Each changed file is encoded with CodeBERT (a mean-pooled, L2-normalized
768-d token embedding) and the commit-level vector is the L2-normalized mean
of its changed-file vectors.  File encodings are cached in a SQLite database
keyed by the content hash, so re-running the stage only encodes new code.
"""

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

DEFAULT_MODEL = "microsoft/codebert-base"
EMBEDDING_DIM = 768


class Encoder(Protocol):
    """Anything that can turn source texts into L2-normalized vectors."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class CodeBERTEncoder:
    """Lazily-loaded CodeBERT mean-pooling encoder (CPU by default)."""

    name: str

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self.device = device
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = self.tokenizer(
                    texts[start : start + self._batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))
        encoded = np.vstack(vectors)
        norms = np.linalg.norm(encoded, axis=1, keepdims=True).clip(min=1e-8)
        return encoded / norms

    _batch_size = 32


def _content_hash(model_name: str, content: str) -> str:
    return hashlib.sha256(f"{model_name}\n{content}".encode()).hexdigest()


class EmbeddingCache:
    """SQLite store mapping content hashes to encoded file vectors."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    def _connect(self) -> None:
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "create table if not exists file_embeddings ("
            " content_hash text primary key,"
            " model text not null,"
            " dim integer not null,"
            " embedding blob not null)"
        )
        self.conn.commit()

    def known(self, hashes: list[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(hashes), 500):
            chunk = hashes[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"select content_hash from file_embeddings where content_hash in ({placeholders})",
                chunk,
            )
            found.update(row[0] for row in rows)
        return found

    def put(self, hashes: list[str], vectors: np.ndarray) -> None:
        self.conn.executemany(
            "insert or ignore into file_embeddings (content_hash, model, dim, embedding) "
            "values (?, ?, ?, ?)",
            [
                (content_hash, self.model_name, int(vectors.shape[1]), vectors[index].tobytes())
                for index, content_hash in enumerate(hashes)
            ],
        )
        self.conn.commit()

    def get(self, hashes: list[str]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 500):
            chunk = hashes[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"select content_hash, dim, embedding from file_embeddings "
                f"where content_hash in ({placeholders})",
                chunk,
            )
            for content_hash, dim, blob in rows:
                result[content_hash] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return result

    model_name = ""

    def close(self) -> None:
        self.conn.close()


@dataclass
class CommitEmbeddings:
    commit_ids: np.ndarray
    embeddings: np.ndarray
    files_used: int
    commits_embedded: int


def encode_sources(
    sources: pd.DataFrame,
    output_root: Path,
    encoder: Encoder,
    cache_db: Path,
    max_length: int = 512,
) -> CommitEmbeddings:
    cache = EmbeddingCache(cache_db)
    cache.model_name = str(getattr(encoder, "name", DEFAULT_MODEL))

    # Deterministic file order per commit so pooling never depends on row order.
    sources = sources.sort_values(["commit_id", "file_path"], kind="stable")
    grouped = sources.groupby("commit_id", sort=True)
    commit_ids = list(grouped.groups.keys())
    file_hashes: list[str] = []
    file_to_commit: list[str] = []
    for commit_id, block in grouped:
        for content in block["content"]:
            file_hashes.append(_content_hash(cache.model_name, str(content)))
            file_to_commit.append(str(commit_id))

    unique_hashes = list(dict.fromkeys(file_hashes))
    cached = cache.known(unique_hashes)
    to_encode_index = [index for index, content_hash in enumerate(unique_hashes) if content_hash not in cached]
    if to_encode_index:
        print(
            f"[embed] encoding {len(to_encode_index)} unique files "
            f"({len(unique_hashes) - len(to_encode_index)} already cached) ...",
            flush=True,
        )
        texts = [str(sources["content"].iloc[index]) for index in to_encode_index]
        vectors = np.asarray(encoder.encode(texts), dtype=np.float32)
        cache.put([unique_hashes[index] for index in to_encode_index], vectors)
    vectors_by_hash = cache.get(unique_hashes)
    cache.close()

    commit_vectors: dict[str, list[np.ndarray]] = {commit_id: [] for commit_id in commit_ids}
    for content_hash, commit_id in zip(file_hashes, file_to_commit, strict=True):
        vector = vectors_by_hash.get(content_hash)
        if vector is not None:
            commit_vectors[commit_id].append(vector)

    embedded_ids: list[str] = []
    embedded_vectors: list[np.ndarray] = []
    files_used = 0
    for commit_id in commit_ids:
        file_vectors = commit_vectors[commit_id]
        if not file_vectors:
            continue
        pooled = np.mean(file_vectors, axis=0)
        pooled = pooled / np.linalg.norm(pooled).clip(min=1e-8)
        embedded_ids.append(commit_id)
        embedded_vectors.append(pooled.astype(np.float32))
        files_used += len(file_vectors)

    output_root.mkdir(parents=True, exist_ok=True)
    # Fixed-width byte strings keep the array loadable with allow_pickle=False.
    width = max((len(value) for value in embedded_ids), default=1)
    np.savez_compressed(
        output_root / "embeddings.npz",
        commit_id=np.array(embedded_ids, dtype=f"S{width}"),
        embeddings=np.vstack(embedded_vectors) if embedded_vectors else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
    )
    (output_root / "embedding_meta.json").write_text(
        json.dumps(
            {
                "model": cache.model_name,
                "embedding_dim": int(EMBEDDING_DIM),
                "max_length": int(max_length),
                "commits_embedded": len(embedded_ids),
                "files_used": files_used,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[embed] embedded {len(embedded_ids)} commits from {files_used} files", flush=True)
    return CommitEmbeddings(
        commit_ids=np.array(embedded_ids),
        embeddings=np.vstack(embedded_vectors) if embedded_vectors else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        files_used=files_used,
        commits_embedded=len(embedded_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True, help="sources.parquet from the extract stage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cache-db", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    sources = pd.read_parquet(args.sources)
    cache_db = args.cache_db or (args.output / "file_cache.sqlite3")
    encoder = CodeBERTEncoder(args.model)
    encode_sources(sources, args.output, encoder, cache_db, max_length=args.max_length)


if __name__ == "__main__":
    main()
