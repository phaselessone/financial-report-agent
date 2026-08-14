from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.retrieval.model_store import ensure_model_downloaded
from src.utils.io import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """将设备选择统一回退:auto 或不可用的 cuda 均落到 cpu。"""
    cuda_available = torch.cuda.is_available()
    if device == "auto":
        if not cuda_available:
            logger.warning("device='auto' but CUDA is unavailable; falling back to CPU.")
            return "cpu"
        return "cuda"
    if device == "cuda" and not cuda_available:
        logger.warning("device='cuda' but torch.cuda.is_available() is False; falling back to CPU.")
        return "cpu"
    return device


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


def _dense_metadata(chunks: list[dict[str, Any]], embedder: "DenseEmbedder", embedding_dim: int) -> dict[str, Any]:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    return {
        "model_name": embedder.model_name,
        "model_path": str(embedder.model_path),
        "embedding_dim": int(embedding_dim),
        "chunk_count": len(chunk_ids),
        "chunk_ids_hash": _chunk_ids_hash(chunk_ids),
        "chunk_text_hash": _chunk_text_hash(chunks),
    }


def _dense_metadata_matches(
    metadata: dict[str, Any],
    *,
    chunks: list[dict[str, Any]],
    embedder: "DenseEmbedder",
    embeddings: np.ndarray,
    chunk_ids: list[str],
) -> bool:
    expected = _dense_metadata(chunks, embedder, embeddings.shape[1])
    return (
        str(metadata.get("model_name", "")) == expected["model_name"]
        and str(metadata.get("model_path", "")) == expected["model_path"]
        and int(metadata.get("embedding_dim", -1)) == expected["embedding_dim"]
        and int(metadata.get("chunk_count", -1)) == expected["chunk_count"] == int(embeddings.shape[0]) == len(chunk_ids)
        and str(metadata.get("chunk_ids_hash", "")) == expected["chunk_ids_hash"] == _chunk_ids_hash(chunk_ids)
        and str(metadata.get("chunk_text_hash", "")) == expected["chunk_text_hash"]
    )


class DenseEmbedder:
    def __init__(self, model_name: str, *, cache_dir: Path, device: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer

        model_path = ensure_model_downloaded(model_name, cache_dir)
        self.model_name = model_name
        self.model_path = model_path
        self.device = resolve_device(device)
        self.model = SentenceTransformer(str(model_path), trust_remote_code=True, device=self.device)

    def encode(self, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], batch_size=1)[0]


def build_or_load_embeddings(
    *,
    chunks: list[dict[str, Any]],
    embedder: DenseEmbedder,
    output_dir: Path,
    rebuild: bool = False,
    batch_size: int = 16,
) -> tuple[np.ndarray, list[str]]:
    ensure_dir(output_dir)
    embeddings_path = output_dir / "embeddings.npy"
    chunk_ids_path = output_dir / "chunk_ids.json"
    metadata_path = output_dir / "dense_metadata.json"

    if not rebuild and embeddings_path.exists() and chunk_ids_path.exists() and metadata_path.exists():
        try:
            embeddings = np.load(embeddings_path).astype(np.float32)
            chunk_ids = list(read_json(chunk_ids_path))
            metadata = read_json(metadata_path)
            if _dense_metadata_matches(
                metadata,
                chunks=chunks,
                embedder=embedder,
                embeddings=embeddings,
                chunk_ids=chunk_ids,
            ):
                return embeddings, chunk_ids
        except Exception as exc:
            logger.warning("Dense embedding cache invalid or unreadable (%s); rebuilding.", exc)

    texts = [chunk["text"] for chunk in chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    embeddings = embedder.encode(texts, batch_size=batch_size)

    tmp_path = embeddings_path.with_name(embeddings_path.stem + ".tmp.npy")
    np.save(tmp_path, embeddings)
    os.replace(tmp_path, embeddings_path)
    write_json(chunk_ids_path, chunk_ids)
    write_json(metadata_path, _dense_metadata(chunks, embedder, embeddings.shape[1]))
    return embeddings, chunk_ids
