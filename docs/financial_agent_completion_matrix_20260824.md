# Financial Report Agent 计划执行与最终验收矩阵

日期：2026-08-29

## 1. 最终结论与状态口径

截至 2026-08-29，代码实现、严格产物合同以及无私有数据的本地 clean-checkout
workflow 已通过；版本控制交付状态以本轮最终 branch/commit 记录为准，不再引用
提交前的瞬时 `git status` 计数。由于 reviewed hard-case/semantic/Stage 6 资产仍
缺失，且没有 hosted Ubuntu/Windows CI 成功记录，项目不能标记为全局
COMPLETE，也不能发布 reviewed benchmark 性能结论。为避免把不同层次的“完成”
混为一谈，本文件只使用以下三种口径：

- **`COMPLETE_LOCAL`：** 代码、接口、测试、严格产物和本地 workflow 合同已经实现并通过当前工作区验收。
- **`SYNTHETIC_CONTRACT_ONLY / NON-PUBLISHABLE`：** 合成 100-case 仅证明四 profile 可以在固定身份和严格产物合同下完整执行；`performance_claim_allowed=false`，不得据此声明真实质量、排名或性能提升。
- **`BLOCKED_EXTERNAL_ASSETS`：** 本地代码路径已经准备好，但缺少不能从 synthetic fixture、模型输出或现有 corpus 推造的 reviewed/private 资产，或缺少外部平台的成功运行证据。

据此，当前总状态是：

- Phase A–D 的生产行为与严格合同为 **`COMPLETE_LOCAL`**。
- Phase E 的四 profile 执行框架、评估与归因框架为 **`COMPLETE_LOCAL`**；当前实际四 profile run 仅为 **`SYNTHETIC_CONTRACT_ONLY / NON-PUBLISHABLE`**。
- Phase F 的 semantic scorer 接线、校准门禁和历史证据门禁为 **`COMPLETE_LOCAL`**；reviewed semantic labels 与 Stage 6 历史 corpus 资产为 **`BLOCKED_EXTERNAL_ASSETS`**。
- Phase G 的本地 CI 合同、文档和可观测性为 **`COMPLETE_LOCAL`**；hosted Ubuntu/Windows CI 成功记录为 **`BLOCKED_EXTERNAL_ASSETS`**。
- 历史 full seed/results 已通过 attestation 身份门禁，但历史 Stage 6 corpus 内容重放仍独立阻塞。
- M10 继续 `KEEP_M10_P2`；现有 corpus 质量证据不支持提升 OCR/Table 优先级。

## 2. 对原 TODO 方案的关键修订

原 TODO 如果直接按 M1→M12 横向推进，会保留多条互相绕开的执行链，并容易把过程指标误写成质量指标。最终实现采用 A→G 顺序，并固定以下约束：

1. `RunIdentity`、严格 trajectory 和原子 EvalBundle 提前到 Phase A；身份不一致的 run 不允许比较。
2. 单跳、多跳、结构化查询、工具和计算统一进入 `ReasoningPlan`，不再保留绕过统一状态机的 multi-hop verified path。
3. `Claim Entailment Yield` 与 `Calculation Execution Success Rate` 只属于 process metrics；只有 reviewed gold 才能计算并命名为 Accuracy。
4. reviewed calculation accuracy 的分母必须同时惩罚漏报、错配、重复和额外预测，不能只在成功匹配的 gold record 上计算。
5. runtime case failure 与 artifact corruption 分离：失败 case 必须持久化失败事件与归因，但不能阻止批次生成可审计 bundle。
6. semantic calibration 只校准 directional `ENTAILED` confidence；不能用同一阈值改变 `CONTRADICTED`。contradiction 在建立独立 reviewed 合同前继续 fail closed。
7. synthetic contract oracle 必须显式开启且永久 non-publishable；reviewed gold evidence 永远 evaluator-only。
8. strict READY 必须来自持久化并与 bundle artifact hashes 绑定的 verdict；live 重算只能把 READY 降为 BLOCKED，不能把无 verdict 的 bundle 临时抬为 READY。
9. corpus identity 必须同时绑定 chunks、structured facts 与 company aliases，不能只哈希 chunks。
10. DuckDB snapshot 通过 read-only source connection 装入内存 FactStore；兼容 legacy/v2 字段且不修改被哈希源文件。
11. historical full seed 身份门禁与 corpus 内容门禁必须分离：chunk ID 存在不代表文本、页码和文档身份相同。
12. baseline profile 不产生 verified claims，因此 evidence-integrity gate 的正确状态是 `NOT_APPLICABLE`，不能为了矩阵整齐而伪造 READY；三个 graph-backed profile 才要求 persisted READY。
13. gold/contract 正确的 must-abstain 必须记为 `EXPECTED_ABSTENTION`，不能计入 `failed_query_count`；被正确停止的重复同参搜索仍保留在 `observed_process_attribution`，供过程诊断但不污染结果失败率。
14. 核心 `SYNTHESIZED` claim 只要来自 `CALCULATE`/`COMPARE` step 或父 claim 已有 calculation lineage，就必须精确绑定成功、已验证且 provenance 完整的 `CalculationRecord`；缺失时不得落入 LLM judge。
15. abstained draft 只能发布 canonical fallback；最终对象必须清空 citation surface，并强制 `support_validation.supported=false`，原 draft 只留在 state/trace 中审计。
16. reviewed 发布门禁必须要求 claim 与 calculation gold denominator 都非零；extra calculation 在 status/operation/evidence/formula/inputs/result/record 七个维度全部计错。
17. claim status 正确但 evidence IDs 错误仍是 reviewed-gold citation failure，必须进入 failure attribution，不能只体现在一个较低的 accuracy 数值里。
18. semantic activation 必须同时绑定 READY readiness、standalone calibration、实际 reviewed labels 文件及其 SHA、local scorer config/identity；所有门禁通过前不得加载模型。
19. historical-full-core/raw 必须在创建 run 目录、加载 corpus 或调用模型前通过正式 Stage 6 evidence/attestation gate，并拒绝空值或 `latest/default/unversioned` model revision。
20. current-dev、strict observability 和四 profile 使用同一套 canonical source-manifest scope；较窄的 smoke 私有哈希口径已移除。

## 3. Phase 完成矩阵

| Phase | 本地实现状态 | 已落地证据 | 外部剩余条件 / 发布限制 |
|---|---|---|---|
| A RunIdentity / strict trace | `COMPLETE_LOCAL` | 15 字段 `RunIdentity`；五文件 EvalBundle；staged validation + atomic replace；比较前身份校验；失败行严格 schema；current-dev real-corpus bundle 为 READY | 无本地代码缺口 |
| B 统一 Reasoning Plan | `COMPLETE_LOCAL` | 单跳、多跳、LOOKUP/SEARCH/CALCULATE/COMPARE 共用 build→plan→execute→observe→coverage→synthesize→claims→verify→finalize；malformed decomposition fail closed | 无本地代码缺口 |
| C Calculation + Claim | `COMPLETE_LOCAL` | evidence-pool operand reuse；单证据多值拒绝静默选首值；Decimal 计算；显式 calculation lineage；atomic ClaimRecord；parent DAG；DERIVED 重算并绑定 `CalculationRecord`；planned comparison synthesis 强制校验 step/result/parent/calculation lineage；canonical abstain；ENTAILED-only finalize | 无本地代码缺口 |
| D Structured Fact v2 | `COMPLETE_LOCAL` | 显式 `MetricSpec`；ACTUAL/ADJUSTED/FORECAST 严格区分；scope/date/revision/basis；Q2 单季与累计口径不再自动混同；DuckDB filter pushdown；legacy JSONL 与 read-only DuckDB 加载 | 无本地代码缺口 |
| E 四 profile / reviewed metrics | 框架 `COMPLETE_LOCAL`；当前 run 为 `SYNTHETIC_CONTRACT_ONLY` | 四个 canonical profile；共享 comparison identity；完整 per-case trajectory；同参 report-search 跨计划去重；case 级 executor failure；gold/process/contract 指标分层；reviewed claim/calculation 非零分母发布门禁；extra calculation 七维惩罚；evidence mismatch 归因；`EXPECTED_ABSTENTION` 与 observed process failure 分离 | 缺正式 reviewed cases/manifest；当前 run `performance_claim_allowed=false`，不得发布排名或质量提升 |
| F semantic / 历史资产 | 代码 `COMPLETE_LOCAL`；资产 `BLOCKED_EXTERNAL_ASSETS` | reviewed label schema；precision-first calibration；local-only directional-NLI scorer；READY readiness + actual labels/hash + calibration + scorer identity 四方绑定；deterministic→NLI→bounded LLM judge；full seed/results 已通过 attestation；historical runtime 前置 Stage 6 门禁；reviewed/historical model revision fail closed | 缺 reviewed semantic labels 及可信 SHA；缺 hash-pinned、内容兼容的 Stage 6 corpus 与 reviewed attestation；正式 semantic activation 未运行 |
| G CI / docs / observability | 本地 `COMPLETE_LOCAL`；hosted CI `BLOCKED_EXTERNAL_ASSETS` | `requirements.lock`；固定 `ruff==0.12.12`；11 个 Phase G/CI/评测核心文件的增量 lint + format-check；Ubuntu/Windows CPU-only workflow；provider-free 四 profile step；263-file clean snapshot 在无 `data/`、无人类 attestation 条件下通过静态门禁、1009 public tests、strict/subset/four-profile/Phase0；持久化 verdict；strict HTML | GitHub token 当前无效，尚无 hosted Ubuntu/Windows green；本地 clean simulation 不能替代 hosted run |
| H M10 OCR/Table | `KEEP_M10_P2` | 3359 页、23942 chunks、194 documents 的 corpus quality gate 仍支持延期 | 只有人工复核证明 OCR/Table 是主要失败来源后才另开计划 |

### 3.1 详细计划的最终 COMPLETE 条件审计

| 详细计划的必要条件 | 当前状态 | 判定依据 |
|---|---|---|
| 单跳与多跳共用受控 Reasoning Plan | `COMPLETE_LOCAL` | 统一 build→plan→execute→observe→coverage→synthesize→claims→verify→finalize 生产图；端到端场景 1–3 通过 |
| 所有 DERIVED 核心 claim 都有 CalculationRecord | `COMPLETE_LOCAL` | `CALCULATE`/`COMPARE` step、claim、parent/depends-on 谱系和成功且 verified/provenance-complete 的 record 强绑定；缺失即 deterministic `INSUFFICIENT` |
| 不存在 LLM 自由算术的 verified path | `COMPLETE_LOCAL` | 缺 operand 不调 LLM 算术；非标准比较措辞也不能绕过 calculation lineage；相关反例回归通过 |
| 所有最终 citation 都来自 ENTAILED claim | `COMPLETE_LOCAL` | claim-specific citation 端到端回归通过；canonical abstain 清空 citation surface |
| evidence-integrity gate 为 PASS/READY | `COMPLETE_LOCAL` | current-dev 和 strict smoke 均有与 bundle hashes 绑定的 persisted `READY`，独立 current-dev 复核仍为 `READY` |
| 四 profile 可真实执行并生成独立 bundle | 执行合同 `COMPLETE_LOCAL`；质量结论 `BLOCKED_EXTERNAL_ASSETS` | 固定 100-case synthetic contract 下四 bundle 已执行；baseline=`NOT_APPLICABLE`，三个 graph-backed profile=`READY`；不允许发布性能结论 |
| 所有比较通过 RunIdentity 校验 | `COMPLETE_LOCAL` | 四 profile 共享 case/benchmark/corpus/model/revision/config/source identity；身份不同的反例在聚合前拒绝 |
| Claim/Calculation Accuracy 来自 reviewed gold | `BLOCKED_EXTERNAL_ASSETS` | evaluator 和非零分母发布门禁已完成；缺 `reviewed_cases.jsonl` / `reviewed_manifest.json` |
| 授权外部资产到位后 readiness=READY | `BLOCKED_EXTERNAL_ASSETS` | 50-row seed/results/attestation 已 READY；顶层仍因 reviewed semantic labels 缺失而 `BLOCKED` |
| semantic scorer 只在 precision-first calibration 通过后启用 | 门禁 `COMPLETE_LOCAL`；正式激活 `BLOCKED_EXTERNAL_ASSETS` | readiness + standalone calibration + actual labels/hash + scorer config/identity 四方 fail-closed 绑定已实现；缺 reviewed labels 与授权 local model/config |
| CI、全量测试、hard-case subset 和文档验收全通过 | 本地 `COMPLETE_LOCAL`；hosted CI `BLOCKED_EXTERNAL_ASSETS` | 全仓、Phase 0、十类 balanced subset、编译、Ruff、依赖和 diff 门禁均通过；`gh` token 失效，无 Ubuntu/Windows hosted green |

结论：所有不依赖外部人工 reviewed/private 资产或 GitHub 平台状态的条件已在本地闭环；由于上表仍有必要条件为 `BLOCKED_EXTERNAL_ASSETS`，整轮升级必须保持非全局 COMPLETE。

## 4. 当前权威验收结果

### 4.1 全仓测试与静态门禁

- clean-checkout public suite：`1009 passed`；RunIdentity/eval 定向组：`59 passed`；evidence/Phase-G 定向组：`31 passed`。
- Phase 0 deterministic smoke：`200 passed`；strict observability：`READY`；balanced hard-case subset：PASS；four-profile contract：`SYNTHETIC_CONTRACT_ONLY`。
- 以上结果来自不含 `data/` 和 human attestation 的 263-file clean snapshot，仅证明本地公开工作流可复现。当前工作区 full-suite JUnit 和最终工件 SHA 在代码冻结后按 2026-08-29 口径重新生成；不得继续引用 2026-08-28 的旧值。
- Balanced hard-case subset：10 cases，10 类各 1 条，subset SHA-256 `5ef1c2dc33de427aeba77a9bf5d564b930dea9e0e5eb07199821f836abfc5dd4`；当前重跑 PASS。
- `uv --no-cache pip check --python .venv\Scripts\python.exe`：通过，`77 packages compatible`。使用 `--no-cache` 是因当前 sandbox 无权读写用户级 `uv` cache，与依赖一致性无关。
- `ruff check`：通过；`ruff format --check`：`11 files already formatted`。该门禁是明确列出的 Phase G/CI/评测核心文件增量作用域，不代表全仓历史样式债已清零。
- `python -m compileall -q src scripts` 与定向 `py_compile`：通过。
- current-dev、strict-observability 和 profile 的最终 RunIdentity/source manifest/hash 以 2026-08-29 代码冻结后的重生工件为准；旧 2026-08-27/28 工件只作为迁移线索。
- `git diff --check`：通过，无 whitespace error；仅有 Windows CRLF 提示。
- 上述通过项证明当前工作区的本地实现与合同一致；它们不替代 reviewed benchmark、semantic calibration、historical corpus replay 或 hosted CI。

### 4.1.1 计划规定的九类端到端场景

冻结代码后按计划逐项运行，参数化展开后共 `12 passed`：

| # | 场景 | 权威回归证据 | 结果 |
|---|---|---|---|
| 1 | 2025 营收：structured shortest path | `ToolAgentFlowTests::test_structured_lookup_is_shortest_path` | PASS |
| 2 | 两期 facts → YoY → DERIVED claim | `ToolAgentFlowTests::test_yoy_fetches_both_facts_then_verifies_derived_claim` | PASS |
| 3 | A/B 四 facts → 两个 YoY → compare → verified conclusion | `test_deterministic_compare_conclusion_overrides_conflicting_model_draft` | PASS；错误模型草稿被确定性结论覆盖，3 条 claims 均 ENTAILED |
| 4 | 报告/页码 claim-specific citation | `ClaimFlowTests::test_report_location_answer_keeps_only_claim_specific_document_and_page` | PASS |
| 5 | 独立子问题 verified partial | `test_independent_multi_hop_success_is_returned_as_verified_partial_answer` | PASS |
| 6 | 缺核心 operand：abstain 且不调用 LLM 计算 | `test_missing_growth_operand_blocks_core_conclusion_and_abstains` | PASS |
| 7 | actual/forecast/adjusted/revision/scope 冲突 fail closed | `test_strict_graph_fails_closed_when_conflicting_facts_lack_a_required_coordinate` | PASS（参数化） |
| 8 | 高相似度但无 directional entailment | `test_high_similarity_without_directional_status_abstains_in_runtime_graph` | PASS |
| 9 | case count/corpus/model 身份不一致拒绝比较 | `test_bundle_group_rejects_identity_mismatch_before_aggregation` | PASS（参数化） |

额外 P0 反例回归也已固定：`test_core_synthesized_compare_claim_without_calculation_cannot_use_llm_judge`、`test_strict_growth_comparison_nonstandard_claim_cannot_bypass_calculation_lineage` 与 `test_strict_finalize_normalizes_abstained_draft_without_publishing_evidence`。它们证明不再存在“缺计算记录但由 LLM 放行比较结论”或“abstain 仍发布未验证正文/引用”的路径。

### 4.2 真实语料 current-dev strict contract

生成命令：

```powershell
.\.venv\Scripts\python.exe scripts\regenerate_dev_trace.py `
  --output-root outputs `
  --overwrite
```

权威结果：

- bundle reference：`outputs/reports/agent_eval_bundle_dev.json`
- bundle reference SHA-256：`7e827a66eac5c5169f376025e2d67dd7bb31660d4814957024cc7a52e91948e3`
- run ID：`run-d229a6cfbe55bc0d1dc0`
- identity hash：`d229a6cfbe55bc0d1dc02768b1cfbecf995a07abf60218e8d8805bc085978c21`
- source manifest hash：`8d76efbc4386a7df97cde585b3454dac47bb2c765cfabcf883f1538cb77576da`
- corpus hash：`0f90e75c99a612d05540a8760b0ae34ba8ebd8e030fe9e881995f97910ce2c69`
- corpus asset manifest hash：`748bf4f60bf24ea02f429cb3c4c9e16de687ccd08e3d9c585f53dd34ad16fabc`
- evidence integrity：`READY`，1 trace、1 claim、1 evidence reference、0 issue、0 error。
- 独立复核报告：`outputs/reports/evidence_integrity_dev_final_20260828.json`，SHA-256 `e695befec84f8727dc875821b4bc22f8d97aa0d64375ed323437424ec843b069`；其 run ID/identity 与 bundle 精确一致。
- 路径：真实 `facts.duckdb` → 两个 structured operands → `net_margin` → DERIVED claim / `CalculationRecord` → deterministic verification → final citation。
- scope：`REAL_CORPUS_DEV_CONTRACT_ONLY`、`publishable=false`、`quality_claims_allowed=false`。
- 旧 `outputs/reports/agent_traces_dev.jsonl` 未覆盖，继续作为 BLOCKED migration fixture。

该 run 证明真实本地 corpus 上的严格计算与证据链闭环，不是 reviewed benchmark 性能证据。

### 4.3 合成 100-case 四 profile 执行合同

运行输出：`outputs/profile_ablation_contract_final_20260827/profile_ablation_summary.json`。

身份与固定清单：

- status：`SYNTHETIC_CONTRACT_ONLY`
- `performance_claim_allowed=false`
- benchmark version：`contract-v1`
- 100 cases，10 类 × 10；其中 10 个 `must_abstain` case 不产生 claim，所以 graph-backed profile 各为 90 claims。
- cases file SHA-256：`32576e9f9acf39b0905841f221ff22ac2bf1a817fe63468aa8f1a801e72fe29f`
- canonical benchmark hash：`44be5c94e0b9c94aaa2c77bfad10352802fff4d6e4b363c76746182825d36de9`
- case IDs hash：`6690f3f46898e6d9ef17462135438d0f4038aaff36cb63c3346275188980f45c`
- manifest：`benchmarks/hard_cases/contract-v1.manifest.json`，SHA-256 `8cce7a6058189bd5dfd383150f8649d324bc92a98b0721bce2faaf5f07ac0bba`
- summary：`outputs/profile_ablation_contract_final_20260827/profile_ablation_summary.json`，SHA-256 `7fac4fc21a8229be69aa393ac291d420198672afb05ac7551b072c79c628e431`
- comparison context hash：`99f5cf9d86393be3486b59c3f3bb9256bad157ac8a9e23edbf39ce140cb4f17d`
- shared source manifest hash：`8d76efbc4386a7df97cde585b3454dac47bb2c765cfabcf883f1538cb77576da`
- source manifest file SHA-256：`c5fe9c67c529c166e186a35c4d332deaa88099437253efb27a32df9ceb3288e9`
- shared corpus hash：`8390c0f6d60e7d03264f39107e59dcbc1203b3657ec7f4552deb96332577a4d9`
- corpus asset manifest hash：`aa32590709786cb924fd599a90160a085da1842a9408b0865e96c4be9083dab8`
- contract oracle chunks SHA-256：`3cd9717312ef8a36e0f509307ad4f1d5bfdb382b78b2f14a75db532f30000086`
- provider/model：`offline-deterministic` / `synthetic-contract-evidence-echo`；无外网、无模型下载、gold 不进入 answerer。

| Profile | Run ID | Evidence integrity | 持久化摘要 |
|---|---|---|---|
| baseline-rag | `run-918290452ab1d7e33b74` | `NOT_APPLICABLE` | baseline 不生成 verified claims；该状态是设计语义，不是缺失 gate |
| agentic-rag | `run-689d75dbbc6dcbc52e2c` | `READY` | 100 rows / 90 claims / 90 entailed evidence refs / 0 issues / 0 errors |
| structured-agent | `run-b27de52d19ea6eeaa301` | `READY` | 100 rows / 90 claims / 90 entailed evidence refs / 0 issues / 0 errors |
| full-agent | `run-37ba68ece49b33e1635f` | `READY` | 100 rows / 90 claims / 90 entailed evidence refs / 0 issues / 0 errors |

四个 identity 已通过可比性校验。每个 profile 的 must-abstain 汇总均为 `failed_query_count=0`、`failure_rate=0.0`、`expected_abstention_count=10`、`observed_process_failure_query_count=10`；这表示正确 abstention 不再被误计为结果失败，同时保留过程诊断。这里的 READY 只表示 graph-backed bundle 的证据引用与持久化工件一致；baseline 的 `NOT_APPLICABLE` 和三个 graph profile 的 READY 都不能把 synthetic contract 变成 reviewed 性能结果。

需要特别区分“矩阵完整执行”和“treatment 被题型触发”：当前
`derived_calculation` 与 `multi_hop` synthetic rows 通过 `SEARCH` 完成且没有
persisted calculation，因此 100-case 结果不能证明 calculation/multi-hop treatment
activation。该行为由 `tests/test_profile_runtime.py` 单独验证；正式效果仍只能由
reviewed benchmark 判定。

### 4.4 Synthetic strict observability contract

- summary：`outputs/strict_observability_smoke_final_20260827/strict_smoke_summary.json`，SHA-256 `4f0f32cb2879265d9d702bf9af870fb11553b5042aceb3370e3f284db49419de`
- bundle reference：`outputs/strict_observability_smoke_final_20260827/strict_smoke_bundle.json`
- run ID：`run-9773d2451bfb44cbcbf8`
- identity hash：`9773d2451bfb44cbcbf85b9844b3d74e4b1235e4765a7850848d0c31f8394366`
- canonical source manifest hash：`8d76efbc4386a7df97cde585b3454dac47bb2c765cfabcf883f1538cb77576da`，与 live/current-dev/profile 完全一致。
- persisted integrity：`READY`，1 trace / 1 claim / 2 entailed evidence refs / 0 issues / 0 errors。
- 结果：真实生产图、YoY `CalculationRecord`、ENTAILED claim、持久化 READY verdict 与 HTML 均通过。
- scope：`SYNTHETIC_CONTRACT_ONLY`、`publishable=false`、`quality_claims_allowed=false`。

### 4.5 Historical full 原件与门禁

2026-08-24 授权恢复后，historical full 的两个原件已按原字节恢复并由独立 attestation 绑定；没有给历史 JSONL 补造逐行 metadata：

- full seed：`data/eval_set/answer_eval_seed_full.jsonl`，50 行，SHA-256 `ecd8fd20cd6beff9a6e4ba9df9b795e97cca1c96a4705d82952489d61c132d21`，状态 READY。
- historical results：`artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl`，50 行，SHA-256 `4940210e2dd6ac8309852581304eb65ad12d445e59b2ae1b892d94c01f35900e`，状态 READY。
- restoration report SHA-256：`fb5235ab42283438451133899eb64b82087db7b53c08a298d0d3b22cea53ab47`。
- attestation：`benchmarks/full/historical-full-raw.attestation.json`，SHA-256 `1ad436865eb25824e3be2c1f6674c98b946df67f0602060b56c9ecc6d9bc97ee`；`check-full-seed` 为 READY。
- 最新 readiness：`outputs/phase0/readiness_authorized_assets_final_20260827.json`，SHA-256 `232c7df821e5d08e8a0004ca3bec9b2618bb730fe03c12bc354a7c02309b3500`；其中 full seed/results 为 READY，顶层仍因 semantic labels 缺失而 BLOCKED。

50-row seed 共涉及 54 个唯一 gold chunk；历史 results 为其中 42 个保留了同 ID 的 snippet/page/file 锚点，另外 12 个只有 reviewed badcase gold 与 attestation 背书。单文档兼容探针虽然复现了历史 doc ID 与 `c0004`/`c0006`，但文本与历史锚点错位，因此：

`historical-full-raw.attestation.json` 当前仅作为本地 reviewed gate 输入；它包含
具体 `attested_by` 人类身份，未获明确签署/发布授权前不得自动 stage 或提交。
其本地 READY 结论不能被解释为该人类 attestation 已获公开发布授权。

- 该 78-chunk 探针只能标记为 `COMPATIBILITY_REBUILD_AUDIT_ONLY`。
- 冻结的兼容重建审计 `outputs/phase0/historical_full_evidence_probe_20260824.json` 状态为 BLOCKED，SHA-256 `8f9e921c1b5ac13019feb78539f93fc2f297efee4ed2de25b9d73d99d065811a`。
- 最新外部门禁 `outputs/phase0/historical_full_evidence_probe_final_20260827.json` 状态仍为 BLOCKED，SHA-256 `54af5173ab68c7eab77e9eb2edf4939d6045899d4c949c2205d0f36932d6749b`；当前明确缺少 Stage 6 chunks snapshot 与 reviewed corpus sidecar。
- 不能把探针复制为 `tmp_stage6_data/chunks/chunks.jsonl`，不能据此消除唯一 pytest skip，也不能修改 reviewed seed。

### 4.6 M10 gate

- decision：`KEEP_M10_P2`
- pages：3359；chunks：23942；documents：194。
- empty-page ratio：0.0042；low-text ratio：0.0473；replacement-char ratio：0；table representation gap：0。
- 最终输出：`outputs/phase0/corpus_quality_gate_final_20260825.json`，SHA-256 `6451fd1375a96732a3735b7d630f3ca92859fe0bcd3e4924f18868abe383e542`。

## 5. 仍被外部资产阻塞的项目

| 阻塞项 | 本地代码是否就绪 | 缺失内容 | 为什么不能本地替代 |
|---|---|---|---|
| Reviewed semantic activation | 是 | `data/eval_set/semantic_verification_labels.jsonl`、可信 labels SHA、通过 precision/coverage 约束的 calibration，以及与其身份一致的本地 directional-NLI model/config | synthetic labels 或模型自标不能替代人工 reviewed labels；未实际激活前不能宣称 semantic 层已验收 |
| Historical Stage 6 replay | 是 | 内容、页码、文档身份均与 historical gold 对应的 hash-pinned `tmp_stage6_data/chunks/chunks.jsonl` | 当前 23942-chunk corpus 与 ID-compatible rebuild 均已证明不能替代历史 snapshot |
| Stage 6 provenance | 是 | `benchmarks/full/historical-stage6-corpus.attestation.json`，需由资产负责人绑定来源、owner、review、chunk count 与 seed/results hashes | observed SHA 只能证明看到某文件，不能证明其来源与 reviewed 身份 |
| Reviewed 四 profile 性能比较 | 是 | `benchmarks/hard_cases/reviewed_cases.jsonl` 与 `reviewed_manifest.json`，以及正式模型的 pinned revision | synthetic 100 只验证执行合同，不能生成可发布的 reviewed Accuracy 或 profile 排名 |
| Hosted CI | 是 | 当前 GitHub token 无效，尚无 Ubuntu/Windows hosted green | 已通过的 263-file clean-checkout simulation 是本地 workflow 证据，但不能替代 GitHub Actions 对最终提交的成功记录 |

这些阻塞项在资产缺失时必须保持 BLOCKED；不得从当前 48-row seed、synthetic 100 cases、兼容重建 corpus 或模型输出来推造。

## 6. 外部资产到位后的正式 reviewed 执行计划

### Step 1：固定授权资产及身份

资产负责人需提供并记录 owner、授权来源、review time、review status 与 SHA-256：

1. reviewed semantic labels 与可信 expected hash；
2. reviewed hard-case cases/manifest；
3. 内容兼容且 hash-pinned 的 Stage 6 chunks snapshot；
4. Stage 6 corpus reviewed attestation；
5. 正式模型的 provider/model/revision 及 local-only directional-NLI runtime config。

### Step 2：运行 readiness 与历史内容门禁

```powershell
$env:SEMANTIC_LABELS_SHA256 = "<authorized-sha256>"

.\.venv\Scripts\python.exe scripts\phase0_gate.py readiness `
  --output outputs\phase0\readiness.json

.\.venv\Scripts\python.exe scripts\historical_full_evidence_gate.py `
  --chunks-path tmp_stage6_data\chunks\chunks.jsonl
```

只有顶层 `status=READY`、semantic calibration 的 `calibrated=true` 且 precision/coverage constraints 全部满足，并且 Stage 6 内容与 attestation 门禁为 READY，才允许继续。不能降低阈值换取通过。

### Step 3：执行正式 reviewed 四 profile

正式命令必须同时提供 reviewed benchmark identity、实际 labels、READY readiness 和已授权 semantic calibration/config；不能只提供 cases 与模型：

```powershell
.\.venv\Scripts\python.exe run_profile_ablation.py `
  --cases-path benchmarks\hard_cases\reviewed_cases.jsonl `
  --reviewed-manifest-path benchmarks\hard_cases\reviewed_manifest.json `
  --reviewed-target-count <declared-release-target> `
  --chunks-path data\chunks\chunks.jsonl `
  --facts-path data\structured\facts.duckdb `
  --company-aliases-path data\structured\company_aliases.json `
  --semantic-calibration-report outputs\phase0\readiness.json `
  --semantic-labels-path data\eval_set\semantic_verification_labels.jsonl `
  --readiness-report outputs\phase0\readiness.json `
  --semantic-scorer-config <authorized-local-directional-nli-runtime-config.json> `
  --llm-provider deepseek `
  --llm-model <model> `
  --llm-model-revision <pinned-revision> `
  --temperature 0 `
  --output-root outputs\profile_ablation_reviewed
```

四 bundle 的 case IDs/count、benchmark/corpus/model/revision/temperature、prompt、budgets、git/source identity 和 semantic activation identity 必须完全一致；只允许 treatment flags 不同。任一 provider/tool error 必须持久化为失败样本，不能终止矩阵或从分母中删除。

### Step 4：最终发布门禁

发布 reviewed 比较结论前必须同时满足：

- baseline bundle 的结构、身份和 case coverage 通过；其 evidence-integrity 状态仍应为 `NOT_APPLICABLE`，因为 baseline 不产生 verified claims。
- agentic-rag、structured-agent、full-agent 三个 graph-backed bundle 均带与 artifact hashes 绑定的 persisted READY verdict。
- reviewed target/category/source quotas 全部满足。
- Claim/Calculation Accuracy 的 reviewed gold denominators 非零，并惩罚所有 extra/missing/mismatched predictions。
- semantic activation identity 与 labels/calibration/config/model hashes 完全一致。
- `historical-full-core` 与 `historical-full-raw` 没有混合聚合。
- historical Stage 6 corpus 内容门禁与 reviewed attestation 均为 READY。
- hosted CI 在 Ubuntu 与 Windows 均实际成功。
- README 或报告中的任何性能数字都直接引用 reviewed bundle，而不是 synthetic/current-dev contract。

在这些条件全部满足前，项目总状态保持 `BLOCKED_EXTERNAL_ASSETS`，不得写成全局 COMPLETE，也不得宣称 reviewed 性能提升。

两个不阻塞本地合同、但仍值得后续硬化的项目是：GitHub Actions action tags 与 lock 中 wheels 尚未做不可变 SHA/hash pin；`offline` 环境变量只证明当前被测路径无需 provider 或模型下载，不是操作系统级网络防火墙。
