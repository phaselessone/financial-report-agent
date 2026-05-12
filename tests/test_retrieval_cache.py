import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import faiss  # noqa: F401
    import rank_bm25  # noqa: F401
except ModuleNotFoundError:
    HAS_RETRIEVAL_DEPS = False
    build_or_load_bm25_index = None
    build_or_load_embeddings = None
    build_or_load_faiss_index = None
else:
    HAS_RETRIEVAL_DEPS = True
    from src.retrieval.bm25_index import build_or_load_bm25_index
    from src.retrieval.embedder import build_or_load_embeddings
    from src.retrieval.faiss_index import build_or_load_faiss_index


class StubEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_path = Path(f"/models/{model_name}")
        self.call_count = 0

    def encode(self, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
        self.call_count += 1
        return np.asarray([[float(index + 1), float(index + 1)] for index, _ in enumerate(texts)], dtype=np.float32)


@unittest.skipUnless(HAS_RETRIEVAL_DEPS, "faiss and rank_bm25 are required for retrieval cache tests")
class RetrievalCacheTests(unittest.TestCase):
    def test_embeddings_rebuild_when_chunk_set_changes(self) -> None:
        chunks_v1 = [{"chunk_id": "c1", "text": "alpha"}]
        chunks_v2 = [{"chunk_id": "c2", "text": "beta"}]
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            embedder_v1 = StubEmbedder("model-a")
            embeddings_v1, chunk_ids_v1 = build_or_load_embeddings(
                chunks=chunks_v1,
                embedder=embedder_v1,
                output_dir=output_dir,
                rebuild=False,
                batch_size=4,
            )
            self.assertEqual(embedder_v1.call_count, 1)
            self.assertEqual(chunk_ids_v1, ["c1"])
            self.assertEqual(embeddings_v1.shape, (1, 2))

            embedder_cached = StubEmbedder("model-a")
            _, cached_chunk_ids = build_or_load_embeddings(
                chunks=chunks_v1,
                embedder=embedder_cached,
                output_dir=output_dir,
                rebuild=False,
                batch_size=4,
            )
            self.assertEqual(embedder_cached.call_count, 0)
            self.assertEqual(cached_chunk_ids, ["c1"])

            embedder_v2 = StubEmbedder("model-a")
            _, chunk_ids_v2 = build_or_load_embeddings(
                chunks=chunks_v2,
                embedder=embedder_v2,
                output_dir=output_dir,
                rebuild=False,
                batch_size=4,
            )
            self.assertEqual(embedder_v2.call_count, 1)
            self.assertEqual(chunk_ids_v2, ["c2"])

    def test_faiss_index_rebuilds_when_chunk_ids_change(self) -> None:
        embeddings_v1 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        embeddings_v2 = np.asarray([[1.0, 0.0]], dtype=np.float32)
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            build_or_load_faiss_index(
                embeddings=embeddings_v1,
                chunk_ids=["c1", "c2"],
                output_dir=output_dir,
                rebuild=False,
            )
            rebuilt = build_or_load_faiss_index(
                embeddings=embeddings_v2,
                chunk_ids=["c3"],
                output_dir=output_dir,
                rebuild=False,
            )
            self.assertEqual(rebuilt.chunk_ids, ["c3"])
            self.assertEqual(rebuilt.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=5)[0][0], "c3")

    def test_bm25_index_rebuilds_when_chunks_change(self) -> None:
        chunks_v1 = [{"chunk_id": "c1", "text": "晶圆代工 景气 上行"}]
        chunks_v2 = [{"chunk_id": "c2", "text": "储能 需求 增长"}]
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            build_or_load_bm25_index(chunks=chunks_v1, output_dir=output_dir, rebuild=False)
            rebuilt = build_or_load_bm25_index(chunks=chunks_v2, output_dir=output_dir, rebuild=False)
            self.assertEqual(rebuilt.chunk_ids, ["c2"])
            self.assertEqual(rebuilt.search("储能", top_k=5)[0][0], "c2")


if __name__ == "__main__":
    unittest.main()
