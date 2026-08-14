from __future__ import annotations

import os
from pathlib import Path

from src.utils.io import ensure_dir
from src.utils.text_utils import safe_model_dir_name


def ensure_model_downloaded(model_name: str, cache_dir: Path, *, local_files_only: bool | None = None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("modelscope is required for model downloads") from exc

    offline = local_files_only if local_files_only is not None else os.environ.get("MODEL_OFFLINE", "") == "1"
    cache_dir = ensure_dir(cache_dir)
    direct_target = cache_dir / model_name
    safe_target = cache_dir / safe_model_dir_name(model_name)
    for local_target in (direct_target, safe_target):
        if local_target.exists() and any(local_target.iterdir()):
            return local_target

    if offline:
        try:
            return Path(snapshot_download(model_name, cache_dir=str(cache_dir), local_files_only=True))
        except Exception as exc:
            raise RuntimeError(
                f"Offline mode is enabled (MODEL_OFFLINE=1 or local_files_only=True) but model "
                f"'{model_name}' is not available in the local cache under '{cache_dir}'. "
                f"Pre-download the model while online, or place it under '{direct_target}' or "
                f"'{safe_target}' before running offline."
            ) from exc

    downloaded_path = Path(snapshot_download(model_name, cache_dir=str(cache_dir)))
    return downloaded_path
