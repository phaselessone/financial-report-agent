from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.io import ensure_dir, read_json, write_json
from src.utils.text_utils import tokenize_for_bm25

logger = logging.getLogger(__name__)


def _chunk_ids_hash(chunk_ids: list[str]) -> str:
    payload = json.dumps(chunk_ids, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _chunk_text_hash(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha1()
    for chunk in chunks:
        digest.update(str(chunk["chunk_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk.get("text", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _bm25_metadata(chunks: list[dict[str, Any]], tokenized_corpus: list[list[str]], chunk_ids: list[str]) -> dict[str, Any]:
    return {
        "chunk_count": len(chunk_ids),
        "chunk_ids_hash": _chunk_ids_hash(chunk_ids),
        "chunk_text_hash": _chunk_text_hash(chunks),
        "tokenized_row_count": len(tokenized_corpus),
    }


def _bm25_metadata_matches(metadata: dict[str, Any], *, chunks: list[dict[str, Any]], chunk_ids: list[str]) -> bool:
    return (
        int(metadata.get("chunk_count", -1)) == len(chunk_ids) == len(chunks)
        and str(metadata.get("chunk_ids_hash", "")) == _chunk_ids_hash(chunk_ids)
        and str(metadata.get("chunk_text_hash", "")) == _chunk_text_hash(chunks)
    )


class BM25ChunkIndex:
    def __init__(self, bm25: BM25Okapi, tokenized_corpus: list[list[str]], chunk_ids: list[str]) -> None:
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunks: list[dict[str, Any]]) -> "BM25ChunkIndex":
        tokenized_corpus = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        return cls(bm25=BM25Okapi(tokenized_corpus), tokenized_corpus=tokenized_corpus, chunk_ids=chunk_ids)

    @classmethod
    def load(cls, output_dir: Path) -> "BM25ChunkIndex":
        with (output_dir / "bm25.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            bm25=payload["bm25"],
            tokenized_corpus=payload["tokenized_corpus"],
            chunk_ids=list(read_json(output_dir / "chunk_ids.json")),
        )

    def save(self, output_dir: Path, *, chunks: list[dict[str, Any]]) -> None:
        ensure_dir(output_dir)
        pkl_path = output_dir / "bm25.pkl"
        tmp_path = pkl_path.with_name(pkl_path.name + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump({"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus}, handle)
        os.replace(tmp_path, pkl_path)
        write_json(output_dir / "chunk_ids.json", self.chunk_ids)
        write_json(output_dir / "bm25_metadata.json", _bm25_metadata(chunks, self.tokenized_corpus, self.chunk_ids))

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[str, float]]:
        query_tokens = tokenize_for_bm25(query)
        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        order = np.argsort(scores)[::-1][:top_k]
        rows: list[tuple[str, float]] = []
        for index in order:
            rows.append((self.chunk_ids[int(index)], float(scores[int(index)])))
        return rows


def build_or_load_bm25_index(
    *,
    chunks: list[dict[str, Any]],
    output_dir: Path,
    rebuild: bool = False,
) -> BM25ChunkIndex:
    metadata_path = output_dir / "bm25_metadata.json"
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if not rebuild and (output_dir / "bm25.pkl").exists() and (output_dir / "chunk_ids.json").exists() and metadata_path.exists():
        try:
            metadata = read_json(metadata_path)
            if _bm25_metadata_matches(metadata, chunks=chunks, chunk_ids=chunk_ids):
                return BM25ChunkIndex.load(output_dir)
        except Exception as exc:
            logger.warning("BM25 index cache invalid or unreadable (%s); rebuilding.", exc)
    bm25_index = BM25ChunkIndex.build(chunks)
    bm25_index.save(output_dir, chunks=chunks)
    return bm25_index


class BM25Retriever:
    def __init__(self, bm25_index: BM25ChunkIndex, chunk_lookup: dict[str, dict[str, Any]]) -> None:
        self.bm25_index = bm25_index
        self.chunk_lookup = chunk_lookup

    def search(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        rows = []
        for rank, (chunk_id, score) in enumerate(self.bm25_index.search(query, top_k=top_k), start=1):
            payload = dict(self.chunk_lookup[chunk_id])
            payload.update({"rank": rank, "score": score, "method": "bm25"})
            rows.append(payload)
        return rows
