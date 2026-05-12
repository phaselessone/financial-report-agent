from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.utils.io import ensure_dir, read_json, write_json


def _chunk_ids_hash(chunk_ids: list[str]) -> str:
    payload = json.dumps(chunk_ids, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _faiss_metadata(chunk_ids: list[str], embedding_dim: int) -> dict[str, Any]:
    return {
        "chunk_count": len(chunk_ids),
        "chunk_ids_hash": _chunk_ids_hash(chunk_ids),
        "embedding_dim": int(embedding_dim),
    }


def _faiss_metadata_matches(metadata: dict[str, Any], *, chunk_ids: list[str], embedding_dim: int) -> bool:
    expected = _faiss_metadata(chunk_ids, embedding_dim)
    return (
        int(metadata.get("chunk_count", -1)) == expected["chunk_count"]
        and str(metadata.get("chunk_ids_hash", "")) == expected["chunk_ids_hash"]
        and int(metadata.get("embedding_dim", -1)) == expected["embedding_dim"]
    )


class FaissDenseIndex:
    def __init__(self, index: faiss.IndexFlatIP, chunk_ids: list[str]) -> None:
        self.index = index
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, embeddings: np.ndarray, chunk_ids: list[str]) -> "FaissDenseIndex":
        embeddings = np.asarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index=index, chunk_ids=chunk_ids)

    @classmethod
    def load(cls, output_dir: Path) -> "FaissDenseIndex":
        index = faiss.read_index(str(output_dir / "faiss.index"))
        chunk_ids = list(read_json(output_dir / "chunk_ids.json"))
        return cls(index=index, chunk_ids=chunk_ids)

    def save(self, output_dir: Path, *, embedding_dim: int) -> None:
        ensure_dir(output_dir)
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        write_json(output_dir / "chunk_ids.json", self.chunk_ids)
        write_json(output_dir / "faiss_metadata.json", _faiss_metadata(self.chunk_ids, embedding_dim))

    def search(self, query_embedding: np.ndarray, *, top_k: int = 20) -> list[tuple[str, float]]:
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(vector, top_k)
        rows: list[tuple[str, float]] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            rows.append((self.chunk_ids[index], float(score)))
        return rows


def build_or_load_faiss_index(
    *,
    embeddings: np.ndarray,
    chunk_ids: list[str],
    output_dir: Path,
    rebuild: bool = False,
) -> FaissDenseIndex:
    index_path = output_dir / "faiss.index"
    chunk_ids_path = output_dir / "chunk_ids.json"
    metadata_path = output_dir / "faiss_metadata.json"
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not rebuild and index_path.exists() and chunk_ids_path.exists() and metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
            if _faiss_metadata_matches(metadata, chunk_ids=chunk_ids, embedding_dim=embeddings.shape[1]):
                return FaissDenseIndex.load(output_dir)
        except Exception:
            pass
    dense_index = FaissDenseIndex.build(embeddings, chunk_ids)
    dense_index.save(output_dir, embedding_dim=embeddings.shape[1])
    return dense_index


class DenseRetriever:
    def __init__(self, dense_index: FaissDenseIndex, chunk_lookup: dict[str, dict[str, Any]]) -> None:
        self.dense_index = dense_index
        self.chunk_lookup = chunk_lookup

    def search(self, query_embedding: np.ndarray, *, top_k: int = 20) -> list[dict[str, Any]]:
        rows = []
        for rank, (chunk_id, score) in enumerate(self.dense_index.search(query_embedding, top_k=top_k), start=1):
            payload = dict(self.chunk_lookup[chunk_id])
            payload.update({"rank": rank, "score": score, "method": "dense"})
            rows.append(payload)
        return rows
