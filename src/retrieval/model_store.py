from __future__ import annotations

from pathlib import Path

from src.utils.io import ensure_dir
from src.utils.text_utils import safe_model_dir_name


def ensure_model_downloaded(model_name: str, cache_dir: Path) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("modelscope is required for model downloads") from exc

    cache_dir = ensure_dir(cache_dir)
    direct_target = cache_dir / model_name
    safe_target = cache_dir / safe_model_dir_name(model_name)
    for local_target in (direct_target, safe_target):
        if local_target.exists() and any(local_target.iterdir()):
            return local_target

    downloaded_path = Path(snapshot_download(model_name, cache_dir=str(cache_dir)))
    return downloaded_path
