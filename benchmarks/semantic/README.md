# Reviewed semantic scorer runtime contract

This directory contains public schemas for a local directional-NLI runtime and
reviewed semantic-label rows. It intentionally does not contain a scorer
config, reviewed label asset, calibration report, model, or model revision.

An authorized runtime config must validate against
`directional-nli-config.schema.json`. The application normalizes all fields and
computes its canonical SHA-256; that identity must exactly match the scorer
identity in the reviewed precision-first calibration report. The runtime always
uses `local_files_only=true` and `trust_remote_code=false`, with evidence as the
premise and the claim as the hypothesis.

Each row must follow `reviewed-semantic-label-v1.schema.json`, use an RFC-3339
review timestamp with timezone, and bind both the claim and evidence via inline
text or SHA-256. Reused claim/evidence IDs may not drift to different content.
The scorer model must be a Hugging Face Hub repo ID, never a local directory or
file path. Its revision must be the canonical lowercase 40-character Git commit
SHA of the locally cached snapshot; branches, tags, abbreviated SHAs, uppercase
hashes, and aliases are rejected. After loading, both tokenizer and model commit
metadata must exactly match that declared revision. Missing or drifting metadata
fails closed.
Calibration requires the existing label/class coverage plus at least five
predicted positives and five true positives; a
near-empty positive prediction set cannot satisfy the precision-first gate.

`run_profile_ablation.py` accepts the Phase 0 readiness JSON through
`--semantic-calibration-report` (using its nested `semantic_calibration` object),
or a standalone calibration report. Semantic activation requires all four
inputs: that report, `--semantic-scorer-config`, the actual reviewed JSONL via
`--semantic-labels-path`, and the top-level READY report via
`--readiness-report`. Before loading the local model, the CLI recomputes the
actual labels SHA-256 and checks labels path/schema, scorer identity, threshold,
calibration identity, and readiness status across all inputs. Missing,
unreviewed, under-precision, identity-mismatched, uncached, or malformed inputs
fail closed before any profile runs.
