# Full 50 Benchmark Provenance

## Source
- Canonical source artifact: `artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl`
- Supporting gold sources:
  - `artifacts/remote_20260401/outputs_badcases/answer_badcases_full.jsonl`
  - `artifacts/remote_20260401/outputs_badcases/abstain_badcases_full.jsonl`
- Canonical restored seed:
  - `data/eval_set/answer_eval_seed_full.jsonl`

## Restoration Workflow
1. Rebuild the historical benchmark corpus from `data/raw_pdfs/incoming` into `tmp_stage6_data/chunks/chunks.jsonl`.
2. Run `scripts/restore_full_answer_seed.py` against the historical result artifact and stage6 chunks to produce:
   - `data/eval_set/answer_eval_seed_full_draft.jsonl`
   - `data/eval_set/answer_eval_seed_full_restore_report.json`
3. Finalize the reviewed draft with `scripts/finalize_full_answer_seed.py` to produce:
   - `data/eval_set/answer_eval_seed_full.jsonl`
   - updated `data/eval_set/answer_eval_seed_full_restore_report.json`

## Source Breakdown
- `artifact_badcase_gold`: 31 rows
- `artifact_abstain_gold`: 5 rows
- `artifact_citation_inferred`: 14 rows

## Review Status
- Review date: `2026-04-05`
- Manual review required rows after finalization: `0`
- Expected full benchmark distribution:
  - `fact = 20`
  - `comparison = 15`
  - `inductive = 15`
  - `must_abstain = 5`

## Notes
- The restored `full 50` benchmark is aligned to the historical `remote_20260401` question IDs.
- The restored seed is intended to be evaluated against the stage6 corpus rooted at `data/raw_pdfs/incoming`, not the earlier four-industry demo corpus under `pdf/`.
