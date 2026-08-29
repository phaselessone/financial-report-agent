import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.run_metadata import (
    build_run_metadata,
    build_source_manifest,
    normalize_benchmark_profile,
)
from src.utils.io import write_jsonl


class RunMetadataTests(unittest.TestCase):
    def test_profile_contract_normalizes_legacy_alias(self) -> None:
        self.assertEqual(normalize_benchmark_profile("historical-full"), "historical-full-raw")

    def test_source_manifest_ignores_generated_python_cache_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "src"
            cache_dir = source_dir / "__pycache__"
            cache_dir.mkdir(parents=True)
            (source_dir / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            cache_path = cache_dir / "app.cpython-312.pyc"
            cache_path.write_bytes(b"first generated cache")

            first = build_source_manifest(cwd=tmp_path)
            cache_path.write_bytes(b"different generated cache")
            second = build_source_manifest(cwd=tmp_path)

            self.assertEqual(first["source_manifest_id"], second["source_manifest_id"])
            self.assertEqual([row["path"] for row in first["files"]], ["src/app.py"])

    def test_source_manifest_binds_the_dependency_lockfile(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "src").mkdir()
            (tmp_path / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
            lock_path = tmp_path / "requirements.lock"
            lock_path.write_text("example==1.0\n", encoding="utf-8")

            first = build_source_manifest(cwd=tmp_path)
            lock_path.write_text("example==2.0\n", encoding="utf-8")
            second = build_source_manifest(cwd=tmp_path)

            self.assertIn("requirements.lock", {row["path"] for row in first["files"]})
            self.assertNotEqual(first["source_manifest_hash"], second["source_manifest_hash"])

    def test_source_manifest_normalizes_checkout_line_endings(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "src"
            source_dir.mkdir()
            source_path = source_dir / "app.py"
            source_path.write_bytes(b"VALUE = 1\nVALUE = 2\n")
            lf_manifest = build_source_manifest(cwd=tmp_path)

            source_path.write_bytes(b"VALUE = 1\r\nVALUE = 2\r\n")
            crlf_manifest = build_source_manifest(cwd=tmp_path)

            self.assertEqual(lf_manifest, crlf_manifest)
            self.assertEqual(lf_manifest["hash_policy"], "text-lf-normalized-v1")
            self.assertEqual(lf_manifest["files"][0]["size_bytes"], len(b"VALUE = 1\nVALUE = 2\n"))

    def test_build_run_metadata_writes_source_manifest_when_git_is_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "src").mkdir()
            (tmp_path / "tests").mkdir()
            (tmp_path / "scripts").mkdir()
            (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (tmp_path / "tests" / "test_app.py").write_text("import unittest\n", encoding="utf-8")
            (tmp_path / "scripts" / "task.py").write_text("print('task')\n", encoding="utf-8")
            (tmp_path / "requirements.txt").write_text("pytest>=8,<9\n", encoding="utf-8")
            chunks_path = tmp_path / "chunks.jsonl"
            benchmark_source_path = tmp_path / "answer_eval_seed_dev.jsonl"
            write_jsonl(chunks_path, [{"chunk_id": "c1", "doc_id": "doc-a", "text": "alpha"}])
            write_jsonl(benchmark_source_path, [{"question_id": "q1", "query": "alpha"}])
            source_manifest_path = tmp_path / "outputs" / "reports" / "source_manifest.json"

            metadata = build_run_metadata(
                chunks_path=chunks_path,
                eval_rows=[{"question_id": "q1", "query": "alpha"}],
                benchmark_source_path=benchmark_source_path,
                benchmark_label="current/dev",
                benchmark_profile="current-dev",
                corpus_label="current",
                split="dev",
                cwd=tmp_path,
                source_manifest_path=source_manifest_path,
            )

            self.assertEqual(metadata["git_sha"], "")
            self.assertTrue(metadata["source_manifest_id"].startswith("source:"))
            self.assertEqual(len(metadata["source_manifest_hash"]), 64)
            self.assertGreaterEqual(metadata["source_file_count"], 4)
            self.assertEqual(metadata["source_manifest_path"], str(source_manifest_path))
            self.assertEqual(
                metadata["benchmark_profile_contract"]["canonical_profiles"],
                ["current-dev", "historical-full-core", "historical-full-raw"],
            )

            manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_manifest_id"], metadata["source_manifest_id"])
            self.assertEqual(manifest["source_manifest_hash"], metadata["source_manifest_hash"])
            manifest_paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("src/app.py", manifest_paths)
            self.assertIn("tests/test_app.py", manifest_paths)
            self.assertIn("scripts/task.py", manifest_paths)
            self.assertIn("requirements.txt", manifest_paths)


if __name__ == "__main__":
    unittest.main()
