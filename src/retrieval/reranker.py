from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.retrieval.embedder import resolve_device
from src.retrieval.model_store import ensure_model_downloaded

# 送入交叉编码器的最大 token 长度(截断阈值)。
RERANK_MAX_LENGTH = 1024


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, cache_dir: Path, device: str = "auto") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = ensure_model_downloaded(model_name, cache_dir)
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), trust_remote_code=True)
        self.device = torch.device(resolve_device(device))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _format_candidate(row: dict[str, Any]) -> str:
        parts: list[str] = []
        if row.get("file_name"):
            parts.append(f"文档: {row['file_name']}")
        if row.get("section_title"):
            parts.append(f"章节: {row['section_title']}")
        if row.get("element_type"):
            parts.append(f"元素: {row['element_type']}")
        if row.get("support_type"):
            parts.append(f"support_type: {row['support_type']}")
        parts.append(f"页码: {row.get('page_start')} - {row.get('page_end')}")
        if row.get("support_span"):
            parts.append(f"精确片段:\n{row['support_span']}")
        if row.get("child_text"):
            parts.append(f"子块:\n{row['child_text']}")
        parts.append(f"父上下文:\n{row['text']}")
        return "\n".join(parts)

    def rerank(self, query: str, rows: list[dict[str, Any]], *, top_k: int = 5, batch_size: int = 8) -> list[dict[str, Any]]:
        if not rows:
            return []

        scored_rows: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [self._format_candidate(row) for row in batch]
            inputs = self.tokenizer(
                [query] * len(batch),
                texts,
                padding=True,
                truncation=True,
                max_length=RERANK_MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
            for row, score in zip(batch, logits, strict=False):
                payload = dict(row)
                payload["rerank_score"] = float(score)
                payload["method"] = "hybrid_rerank"
                scored_rows.append(payload)

        results = sorted(scored_rows, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
        for rank, row in enumerate(results, start=1):
            row["rank"] = rank
            row["score"] = row["rerank_score"]
        return results
