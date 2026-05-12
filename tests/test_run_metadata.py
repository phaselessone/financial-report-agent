import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.run_metadata import build_run_metadata
from src.utils.io import write_jsonl


class RunMetadataTests(unittest.TestCase):
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
            self.assertGreaterEqual(metadata["source_file_count"], 4)
            self.assertEqual(metadata["source_manifest_path"], str(source_manifest_path))

            manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_manifest_id"], metadata["source_manifest_id"])
            manifest_paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("src/app.py", manifest_paths)
            self.assertIn("tests/test_app.py", manifest_paths)
            self.assertIn("scripts/task.py", manifest_paths)
            self.assertIn("requirements.txt", manifest_paths)


if __name__ == "__main__":
    unittest.main()
