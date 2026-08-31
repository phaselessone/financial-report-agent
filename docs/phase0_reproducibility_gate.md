# Phase 0 可复现基线与资产门禁

本门禁把公开仓库可以确定性重跑的测试，与必须由仓库外恢复的历史 full benchmark 明确分开。它不会跳过、修改或降低 `tests/test_full_seed_restore.py` 的标准。

## 确定性 P0 smoke

以下入口在 Windows PowerShell、cmd、macOS 和 Linux 上均通过 Python 调用，不依赖 Bash、模型下载、API key 或私有数据：

```text
python scripts/phase0_gate.py smoke
```

使用虚拟环境时，将 `python` 替换为 `.venv\Scripts\python.exe`（Windows）或 `.venv/bin/python`（POSIX）。也可运行 `make p0-smoke`。该入口以显式测试文件覆盖 claim、calculator、agent flow、structured fact、trajectory、evidence integrity/semantic calibration 和 frozen baseline regression，任何失败均原样返回非零退出码。

## 基线采集

```text
python scripts/phase0_gate.py baseline
```

默认生成 `outputs/phase0/baseline.json`。`outputs/` 已被 Git 忽略，不提交生成报告。报告记录：Git commit/branch/dirty 状态、Python 与平台、直接依赖的已安装版本、requirements/runtime config 哈希、非敏感模型及运行时配置、标准测试命令和输出位置。脚本只读取允许的非敏感环境变量，不读取或记录 API key、base URL、`.env*` 内容。

若本机已恢复历史 full benchmark，可加 `--include-full-seed`；此选项会先执行严格资产检查，缺失或不一致时不会生成一个伪完整基线。

## 私有 full benchmark 资产

规范路径为：

- reviewed 50-row seed：`data/eval_set/answer_eval_seed_full.jsonl`
- historical result identity source：`artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl`
- reviewed asset attestation：`benchmarks/full/historical-full-raw.attestation.json`
- reviewed restoration evidence：`data/eval_set/answer_eval_seed_full_restore_report.json`

恢复资产及其授权 attestation 后运行：

```text
python scripts/phase0_gate.py check-full-seed
```

检查要求两个 JSONL 原件和 attestation 都存在。Attestation 必须把精确的资产路径、50 行计数和 SHA-256 绑定到 `historical-full-raw`，记录正向 reviewed status、custodian、batch、review time，并绑定 provenance 文档及 reviewed restoration report；恢复报告必须仍为 50 条、fact/comparison/inductive = 20/15/15、固定五条 abstain、待人工复核为 0。门禁随后重新计算两个原件及恢复报告的哈希，并校验 seed/results question IDs 完全一致。这样不需要给 2026-04-01 的历史结果逐行写入后来新增的 metadata，也不会为了通过新 schema 改写历史原件。

兼容模式仍允许对“本身已带逐行审阅 metadata”的新资产使用 `FULL_SEED_SHA256` 和 `FULL_RESULTS_SHA256`。若同时提供 sidecar 与环境变量，两组哈希必须完全一致，否则 fail closed。缺失资产、证据、可信哈希或审阅身份均返回退出码 2；`answer_eval_seed_current_full.jsonl` 不能替代历史 full seed。

### 历史 Stage 6 证据内容门禁

`check-full-seed` 证明 50 行 benchmark 原件及其审阅链身份，不证明某个重新 ingest
的 `chunks.jsonl` 与历史 gold evidence 内容相同。正式重放 historical full 前还必须运行：

```text
python scripts/historical_full_evidence_gate.py \
  --chunks-path tmp_stage6_data/chunks/chunks.jsonl
```

正式门禁要求同时恢复
`benchmarks/full/historical-stage6-corpus.attestation.json`；其合同见
`benchmarks/full/historical-stage6-corpus.attestation.schema.json`。Sidecar 必须绑定
固定历史身份 `corpus:4ff7ba48fd38`、标准路径
`tmp_stage6_data/chunks/chunks.jsonl`、历史整文件 SHA-1
`4ff7ba48fd38a4400e581fac32c43c4217de1ece`、当前实算 SHA-256、chunk 数、
来源、owner、取得时间、source ref、正向 review 身份，以及当前 seed/results 的精确
SHA-256。这里的 SHA-1 只用于与历史运行 metadata 对齐，安全完整性仍由 SHA-256
承担。Validator 会拒绝 alternate path、未知字段和缺少时区的 review/source 时间。
单独设置 `STAGE6_CHUNKS_SHA256` 只允许生成 audit-only 诊断报告，不能令正式门禁
READY；即使内容匹配，也只返回 `DIAGNOSTIC_CONTENT_COMPATIBLE` /
`DIAGNOSTIC_ONLY` 并保持退出码 2。

该门禁重新计算候选 corpus SHA-256，要求全部 54 个唯一 gold chunk 精确存在，并对
历史结果中有锚点的 42 个 chunk 同时校验文本、页码、doc ID 与文件名。其余 12 个
来自 reviewed badcase gold、但历史结果未保留相同 chunk snippet 的条目，只能在 full
attestation 已验证、目标文档身份一致时记为 `ATTESTED_UNANCHORED`；它们不会被写成
content-verified。任一 ID 缺失、文本/页码错位、重复 chunk ID、未固定 hash、缺少
corpus provenance/review sidecar 或 attestation 失败都会得到 `BLOCKED` 和退出码 2。

这项检查修复了旧 materialization 测试的盲点：只要 ID 可解析就得到 50/50，仍可能
引用到同一 ordinal 下的不同文本。现在当 `tmp_stage6_data/chunks/chunks.jsonl` 存在时，
`tests/test_full_seed_restore.py` 还要求 `STAGE6_CHUNKS_SHA256`，并先通过内容门禁再做
materialization。缺失外部 snapshot 时该测试仍明确 skip，不能据此宣称完整回归通过。

资产门禁成功后，运行原始完整测试：

```text
python -m pytest -q
```

没有私有资产时，可运行确定性公开套件 `python -m pytest -q --ignore=tests/test_full_seed_restore.py`，但报告必须标注 full fixture 缺失，不得称为完整测试全绿。

## Readiness 与 benchmark profile

三类 benchmark profile 必须在报告、trace metadata 和 artifact 目录中保持一致：

- `current-dev`：当前开发集，只允许 `--split dev`。
- `historical-full-core`：历史 full 的去冲突核心口径。
- `historical-full-raw`：历史 full 原始口径，保留冲突组。

`historical-full` 仅是兼容旧命令的显式别名，规范化后等同于
`historical-full-raw`。不同 profile 的指标不得直接合并。

在声称 full benchmark 或 semantic calibration 可用前，先运行：

```text
SEMANTIC_LABELS_SHA256=<authorized-sha256> \
python scripts/phase0_gate.py readiness
```

该命令写入 `outputs/phase0/readiness.json`，逐项记录 full seed、历史结果和
reviewed semantic labels 的路径、SHA-256、行数与阻塞原因。Full seed/results 的可信哈希默认来自 reviewed attestation；语义标签的预期哈希仍必须来自其授权发布清单。Semantic labels 还必须绑定单一 directional-NLI scorer 的 kind/model/revision/config hash，至少 20 条且 ENTAILED/非 ENTAILED 各至少 5 条，并满足 precision-first constraint。每行 claim/evidence 必须有正文或 SHA-256 内容绑定，reviewed_at 必须带时区，scorer revision 不得是 moving branch；校准阈值还必须至少覆盖 5 个 predicted positives 和 5 个 true positives。任一资产、review contract、profile identity、coverage 或 calibration 不通过时，顶层状态为 `BLOCKED`、退出码为 2；当前 48 行 seed 不会被替代使用。

已有 agent trace 还必须通过 evidence integrity gate：

```text
python scripts/evidence_integrity_gate.py \
  --bundle-path outputs/reports/agent_eval_bundle_dev.json \
  --chunks-path data/chunks/chunks.jsonl
```

该 gate 检查 claim/status/evidence/citation/calculation/final-answer 的可解析性和
来源完整性；存在任何问题时返回 `BLOCKED` 和退出码 2。
