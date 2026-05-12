from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        ensure_dir(path.parent)
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write_row(self, row: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
