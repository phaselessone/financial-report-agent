import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evaluation.eval_bundle import read_eval_bundle_reference, write_eval_bundle, write_eval_bundle_reference
from src.evaluation.output_archive import archive_output_bundle, resolve_artifact_dir, write_scratch_run_policy
from src.evaluation.run_identity import RunIdentity, hash_case_ids


class OutputArchiveTests(unittest.TestCase):
    @staticmethod
    def _identity() -> RunIdentity:
        return RunIdentity(
            benchmark_profile="current-dev",
            benchmark_version="v1",
            benchmark_hash="benchmark-hash",
            case_ids_hash=hash_case_ids(["q1"]),
            case_count=1,
            corpus_hash="corpus-hash",
            model="model",
            provider="provider",
            model_revision="revision",
            temperature=0.0,
            prompt_version="prompt-v1",
            budgets={"max_steps": 1},
            feature_flags={"runtime": {}, "treatments": {"pipeline": "baseline"}},
            git_commit="commit",
            source_manifest_hash="manifest-hash",
        )

    def test_resolve_artifact_dir_avoids_collisions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            existing = tmp_path / "remote_20260410_dev_current-dev"
            existing.mkdir()
            artifact_dir = resolve_artifact_dir(
                artifact_root=tmp_path,
                artifact_label="remote_20260410_dev_current-dev",
            )
            self.assertEqual(artifact_dir.name, "remote_20260410_dev_current-dev-2")

    def test_archive_output_bundle_copies_reports_and_badcases(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            reports_dir = output_dir / "reports"
            badcases_dir = output_dir / "badcases"
            reports_dir.mkdir(parents=True)
            badcases_dir.mkdir(parents=True)
            (reports_dir / "answer_eval_summary_dev.json").write_text('{"ok": true}', encoding="utf-8")
            (badcases_dir / "answer_badcases_dev.jsonl").write_text("{}", encoding="utf-8")

            artifact_dir = tmp_path / "artifacts" / "remote_20260410_dev_current-dev"
            archive_output_bundle(output_dir=output_dir, artifact_dir=artifact_dir)

            self.assertTrue((artifact_dir / "reports" / "answer_eval_summary_dev.json").exists())
            self.assertTrue((artifact_dir / "badcases" / "answer_badcases_dev.jsonl").exists())

    def test_archived_eval_bundle_reference_is_self_contained_and_relocatable(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            bundle = write_eval_bundle(
                output_dir / "eval",
                identity=self._identity(),
                config={},
                metrics={},
                per_case=[{"question_id": "q1"}],
                trajectories=[],
                metadata={"mode": "baseline"},
            )
            reference = output_dir / "reports" / "agent_eval_bundle_dev.json"
            write_eval_bundle_reference(reference, bundle)
            artifact_dir = tmp_path / "artifacts" / "run"
            archive_output_bundle(output_dir=output_dir, artifact_dir=artifact_dir)

            relocated = tmp_path / "relocated"
            shutil.copytree(artifact_dir, relocated)
            shutil.rmtree(output_dir)
            shutil.rmtree(artifact_dir)

            relocated_reference = relocated / "reports" / reference.name
            reference_row = json.loads(relocated_reference.read_text(encoding="utf-8"))
            loaded = read_eval_bundle_reference(relocated_reference)
            self.assertFalse(Path(reference_row["bundle_path"]).is_absolute())
            self.assertEqual(loaded.path.resolve(), (relocated / "eval" / bundle.identity.run_id).resolve())

    def test_write_scratch_run_policy_records_last_archive(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "outputs"
            output_dir.mkdir()
            summary_path = output_dir / "reports" / "answer_eval_summary_dev.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}", encoding="utf-8")
            source_manifest_path = output_dir / "reports" / "source_manifest.json"
            source_manifest_path.write_text("{}", encoding="utf-8")
            artifact_dir = tmp_path / "artifacts" / "remote_20260410_dev_current-dev"
            artifact_dir.mkdir(parents=True)

            policy_path = write_scratch_run_policy(
                output_dir=output_dir,
                artifact_dir=artifact_dir,
                split="dev",
                benchmark_profile="current-dev",
                summary_path=summary_path,
                source_manifest_path=source_manifest_path,
            )

            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertEqual(policy["mode"], "scratch")
            self.assertEqual(policy["split"], "dev")
            self.assertEqual(policy["benchmark_profile"], "current-dev")
            self.assertEqual(policy["last_archived_run"], str(artifact_dir))
            self.assertEqual(policy["latest_source_manifest_path"], str(source_manifest_path))

    def test_write_scratch_run_policy_keeps_profile_separate_from_treatments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            output_dir.mkdir()

            policy_path = write_scratch_run_policy(
                output_dir=output_dir,
                artifact_dir=None,
                split="dev",
                benchmark_profile="current-dev",
                treatments={"pipeline": "agentic", "tool_orchestration": True},
                summary_path=output_dir / "reports" / "summary.json",
            )

            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertEqual(policy["benchmark_profile"], "current-dev")
            self.assertEqual(
                policy["treatments"],
                {"pipeline": "agentic", "tool_orchestration": True},
            )


if __name__ == "__main__":
    unittest.main()
