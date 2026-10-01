"""Generate CodeBERT file-level embeddings from the file-level Parquet dataset.

Produces one embedding vector per (commit_id, file_path) pair, stored in an
NPZ file keyed by a stable hash of (commit_id, file_path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "microsoft/codebert-base"
EMBEDDING_DIM = 768


def _row_key(commit_id: str, file_path: str) -> str:
    return hashlib.sha256(f"{commit_id}\x00{file_path}".encode()).hexdigest()[:32]


def embed_file_level(
    files_parquet: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    cache_db: Path | None = None,
) -> None:
    """Embed all files in files_parquet and write file_embeddings.npz."""
    from ml.embeddings.embed import CodeBERTEncoder, EmbeddingCache, _content_hash

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_db or (output_dir / "file_cache.sqlite3")
    frame = pd.read_parquet(files_parquet, columns=["commit_id", "file_path", "content"])
    encoder = CodeBERTEncoder(model_name)
    cache = EmbeddingCache(cache_path, model_name=model_name)

    content_hashes = [
        _content_hash(model_name, str(row.content))
        for row in frame.itertuples(index=False)
    ]
    hash_to_content: dict[str, str] = {}
    for h, row in zip(content_hashes, frame.itertuples(index=False)):
        if h not in hash_to_content:
            hash_to_content[h] = str(row.content)

    unique_hashes = list(dict.fromkeys(content_hashes))
    cached = cache.known(unique_hashes)
    to_encode = [h for h in unique_hashes if h not in cached]
    if to_encode:
        texts = [hash_to_content[h] for h in to_encode]
        vectors = np.asarray(encoder.encode(texts), dtype=np.float32)
        cache.put(to_encode, vectors)

    vectors_by_hash = cache.get(unique_hashes)
    cache.close()

    keys: list[str] = []
    embeddings: list[np.ndarray] = []
    for content_hash, row in zip(content_hashes, frame.itertuples(index=False)):
        vec = vectors_by_hash.get(content_hash)
        if vec is not None:
            keys.append(_row_key(str(row.commit_id), str(row.file_path)))
            embeddings.append(vec)

    width = max((len(k) for k in keys), default=1)
    np.savez_compressed(
        output_dir / "file_embeddings.npz",
        keys=np.array(keys, dtype=f"S{width}"),
        embeddings=(
            np.vstack(embeddings)
            if embeddings
            else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        ),
    )
    (output_dir / "file_embedding_meta.json").write_text(
        json.dumps({"model": model_name, "dim": EMBEDDING_DIM, "files": len(keys)}),
        encoding="utf-8",
    )
    print(f"[file_embeddings] embedded {len(keys)} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cache-db", type=Path, default=None)
    args = parser.parse_args()
    embed_file_level(args.files_parquet, args.output, args.model, args.cache_db)


if __name__ == "__main__":
    main()
