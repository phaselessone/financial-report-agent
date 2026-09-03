# Agent observability demo

`run_agent_eval.py --mode agentic` writes a strict evaluation bundle and a
small reference at `outputs/reports/agent_eval_bundle_dev.json`. The
observability CLI reads that bundle, verifies its hashes and RunIdentity,
runs the evidence-integrity gate against the corpus chunks, and renders one
trajectory as a standalone HTML review page. It does not load a model or
contact a provider.

To create the repository's current, real-asset-bound dev reference without a
provider, run:

```powershell
.\.venv\Scripts\python.exe scripts\regenerate_dev_trace.py --output-root outputs
```

This uses the current-local chunks, structured facts and company aliases, forces
one deterministic structured `net_margin` path, and independently rechecks the
persisted bundle. It is permanently marked `REAL_CORPUS_DEV_CONTRACT_ONLY` and
non-publishable. The older `agent_traces_dev.jsonl` remains an intentionally
BLOCKED migration fixture.

```powershell
.\.venv\Scripts\python.exe scripts\observability_demo.py `
  --bundle-path outputs\reports\agent_eval_bundle_dev.json `
  --chunks-path data\chunks\chunks.jsonl `
  --question-id q1 `
  --output outputs\observability\q1.html
```

`--bundle-path` accepts either the strict bundle directory or its reference
JSON. If omitted, it defaults to the reference above. Missing
`--question-id` selects the first trajectory in source order.

## What `READY` means

The CLI exits 0 and displays `READY` only when all of the following hold:

- the five bundle files and their artifact hashes pass strict bundle loading;
- agentic metadata contains a persisted, bundle-bound `READY` integrity verdict
  whose run ID, identity hash, case-ID hash, and artifact hashes match this exact
  bundle;
- the selected trajectory carries the bundle's exact `run_id` and
  `run_identity`;
- claim states, evidence IDs, citations, document/page metadata, calculation
  provenance, and final-answer evidence pass the evidence-integrity gate.

The renderer also performs a live evidence/chunk recheck. That live recheck can
demote a previously validated bundle to `BLOCKED`, but a live recheck cannot promote
an agentic bundle that lacks its persisted verdict to `READY`. Such a bundle is
rendered as `BLOCKED` and the CLI exits 2.

This is an integrity/process result, not a reviewed answer-quality score. A
synthetic contract run cannot support a performance-improvement claim.

## Provider-free real-graph smoke

The repository includes a deterministic end-to-end producer for reviewing the
strict contract without a corpus, API key, network request, or model download:

```powershell
.\.venv\Scripts\python.exe scripts\strict_observability_smoke.py `
  --output-root outputs\strict_observability_smoke `
  --overwrite
```

Unlike a hand-authored trace fixture, this command invokes the production agent
graph, materializes its claim verification and node trajectory, writes the
five-file strict EvalBundle through the production integrity writer, persists
the READY verdict, writes a relative/movable bundle reference, and then invokes
the normal HTML renderer. The generated `strict_smoke_summary.json` explicitly
records `SYNTHETIC_CONTRACT_ONLY`, `publishable=false`, and that no external
network/model download was used. READY here proves structural and provenance
integrity only; it is not reviewed benchmark evidence.

The page shows the question and answer/abstention, every claim and status,
ENTAILED-only final citations, calculation records, reasoning/tool/dependency
steps, failure attribution, prompt/completion/total tokens, API/end-to-end
latency, and the exact RunIdentity. Rows without citations, calculations, or
failures render explicit empty records rather than invented content.

## Legacy traces are never promoted

For forensic inspection only, `--trace-path` accepts the old standalone JSONL:

```powershell
.\.venv\Scripts\python.exe scripts\observability_demo.py `
  --trace-path outputs\reports\agent_traces_dev.jsonl `
  --output outputs\observability\legacy_trace.html
```

That mode always labels the source `legacy_trace_unverified`, displays
`BLOCKED`, and exits 2 because a raw JSONL has no immutable bundle identity or
artifact hashes. It is never wrapped or reported as a passing strict run.

The demo is deliberately static and local. It is not an application server and
does not add authentication, user accounts, billing, real-time market data, or
trading actions.
