# Hard-case benchmark (M4/M5)

This directory ships **100 deterministic synthetic contract cases**, ten in each of ten categories. Every row in `cases.jsonl` is labelled `synthetic: true` and `review_status: synthetic_not_human_reviewed`; `schema.json` is its row schema. These fixtures test routing, disambiguation, calculation, abstention, citation and trajectory behavior. They are not human-reviewed financial evidence and cannot support a quality or profile-performance claim.

`contract-v1.manifest.json` pins the exact file SHA-256, canonical benchmark
hash, case-ID hash, row count, and 10×10 category distribution. The balanced
subset CI gate validates this manifest before sampling; schema-compatible
content drift therefore fails instead of silently replacing the contract.

## Reviewed asset contract

No real reviewed benchmark is included in this repository. An external reviewed release is expected at `reviewed_cases.jsonl` with a sibling `reviewed_manifest.json`, or at explicit paths passed to `load_reviewed_cases`. Their contracts are `reviewed-v1.schema.json` and `reviewed-manifest.schema.json`.

The evaluator recomputes two order-independent identities: a canonical cases hash (with self-reported row hash fields removed) and a canonical case-ID hash. Both must equal the manifest. A row-level `benchmark_hash` is only an optional legacy annotation and is never the benchmark identity authority.

Each evidence record has a `source_id`. It must resolve through the row's `source_provenance`, and that provenance must match the manifest entry's source hash and locator. The manifest declares category and source quotas; validation recomputes counts and any shortfall from the cases rather than trusting a reported coverage object.

Formal publication additionally requires every manifest source to declare a
relative `path`, an explicit `--reviewed-source-root`, and a human
`release_attestation` (`status=reviewed`, reviewer, batch, and RFC-3339 timestamp
with timezone). The validator confines each resolved path beneath that root and
recomputes the source file SHA-256 before any model is loaded. Absolute paths,
path traversal, missing files, hash drift, invalid timestamps, and input
manifests that self-report `source_verification` all fail closed. A legacy
structural manifest can still be parsed for migration, but it cannot pass the
reviewed publication gate.

Gold calculation requirements depend on status. `SUCCESS` requires operands and a result; `FAILED`, `INSUFFICIENT`, `AMBIGUOUS` and `SKIPPED` may omit them. Metrics score only applicable gold dimensions. Process metrics use only the run's own trace/prediction data, while reviewed gold metrics are emitted separately. Every rate reports `numerator`, `denominator`, `micro` and `macro`.

`Unnecessary Tool Call Rate` is a call-event metric: its denominator is every
actual tool call, and its numerator is every call whose tool type is absent
from the case's gold `required_tools`. Consequently, any call is unnecessary
when `required_tools` is empty. The schema specifies required tool types, not
per-tool call counts, so repeated calls to a required tool are not classified
as unnecessary by this metric; repeated calls to a non-required tool are
counted once per invocation. `micro` is the primary aggregate across call
events, `macro` is the mean of per-case rates, and repeated-call cost is also
visible independently in `Avg Tool Calls`.

`performance_claim_allowed` is true only when all of the following hold: the reviewed manifest validates; source files were recomputed under the confined root; the release attestation validates; a positive release target is explicitly supplied and met; all ten categories have positive declared quotas and meet them; every manifest source has a positive quota and meets it; every case has a prediction; and the reviewed Claim Verification Accuracy and Calculation Record Accuracy both have non-zero gold denominators. Otherwise the result remains `COVERAGE_ONLY` and `publish_gate.blocking_reasons` records why. Unexpected calculation predictions count as false in every reviewed calculation dimension. Synthetic input is always `SYNTHETIC_CONTRACT_ONLY`.

The four executable profile IDs are `baseline-rag`, `agentic-rag`, `structured-agent` and `full-agent`. Runtime treatments come from `src.evaluation.profile_runtime.PROFILE_SPECS`; the compatibility table in the hard-case evaluator is derived from that registry. Atomic claim extraction and deterministic provenance verification are a non-optional graph integrity invariant (`strict_claim_provenance`), including for `structured-agent`; the `claim_verification` treatment controls enhanced verification and recovery only.

## Formal four-profile command

```powershell
.\.venv\Scripts\python.exe run_profile_ablation.py `
  --cases-path <reviewed_cases.jsonl> `
  --reviewed-manifest-path <reviewed_manifest.json> `
  --reviewed-source-root <reviewed_source_root> `
  --reviewed-target-count <release_target> `
  --chunks-path <shared_corpus_chunks.jsonl> `
  --facts-path <facts.duckdb> `
  --company-aliases-path <company_aliases.json> `
  --llm-provider deepseek `
  --llm-model <model> `
  --llm-model-revision <revision> `
  --output-root outputs/profile_ablation
```

The CLI rejects synthetic input by default. Reviewed gold evidence is never used as retrieval input; every formal profile uses the shared corpus supplied by `--chunks-path`. Contract-only execution must opt in with both `--allow-synthetic-contract` and `--cases-path benchmarks/hard_cases/cases.jsonl`; its publication status remains `SYNTHETIC_CONTRACT_ONLY`. The additional `--contract-oracle` flag may expose synthetic case evidence to retrieval for contract checks only; the RunIdentity records `synthetic-contract-oracle` and the result can never become publishable.

For a provider-free execution and bundle contract, add `--offline-contract` or
run `make four-profile-contract`. This still traverses each profile executor for
all 100 cases and writes four independent bundles, but the deterministic
answerer only echoes retrieved synthetic evidence and never reads gold answers.
It is an execution/plumbing check, not category-quality evidence. The resulting
matrix always records `performance_claim_allowed=false`.

In this exact offline-oracle mode, `run_profile_ablation.py` deterministically
materializes a minimal public-safe financial fact and company-alias file under
the selected output root. The generated chunks, facts, and aliases are the
actual three assets hashed into the corpus manifest and RunIdentity, so the
contract runs in a clean checkout with no ignored `data/` directory. Other
modes continue to require the caller's explicit corpus assets.
