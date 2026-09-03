# Financial Report Agent 计划执行与最终验收矩阵

日期：2026-09-03

## 1. 最终结论与状态口径

截至 2026-09-03，代码实现、严格产物合同与本轮资产门禁加固已通过本地完整 suite
和 CI 等价命令。上一冻结基线 `446a316` 的 hosted deterministic-ci run
[`33260006502`](https://github.com/phaselessone/financial-report-agent/actions/runs/33260006502)
已在 Ubuntu 与 Windows 通过；本轮 2026-09-03 hardening 在本快照中尚待提交并触发
新的 hosted run，因此不能用旧 run 冒充新代码的 hosted 证据。reviewed
hard-case/semantic/Stage 6 资产仍缺失，
因此项目不能标记为全局 COMPLETE，也不能发布 reviewed benchmark 性能结论。为避免把不同层次的“完成”
混为一谈，本文件只使用以下三种口径：

- **`COMPLETE_LOCAL`：** 代码、接口、测试、严格产物和本地 workflow 合同已经实现并通过当前工作区验收。
- **`SYNTHETIC_CONTRACT_ONLY / NON-PUBLISHABLE`：** 合成 100-case 仅证明四 profile 可以在固定身份和严格产物合同下完整执行；`performance_claim_allowed=false`，不得据此声明真实质量、排名或性能提升。
- **`BLOCKED_EXTERNAL_ASSETS`：** 本地代码路径已经准备好，但缺少不能从 synthetic fixture、模型输出或现有 corpus 推造的 reviewed/private 资产。

据此，当前总状态是：

- Phase A–D 的生产行为与严格合同为 **`COMPLETE_LOCAL`**。
- Phase E 的四 profile 执行框架、评估与归因框架为 **`COMPLETE_LOCAL`**；当前实际四 profile run 仅为 **`SYNTHETIC_CONTRACT_ONLY / NON-PUBLISHABLE`**。
- Phase F 的 semantic scorer 接线、校准门禁和历史证据门禁为 **`COMPLETE_LOCAL`**；reviewed semantic labels 与 Stage 6 历史 corpus 资产为 **`BLOCKED_EXTERNAL_ASSETS`**。
- Phase G 的 CI 合同、文档和可观测性为 **`COMPLETE_LOCAL`**；上一冻结基线为 **`HOSTED_GREEN`**，本轮 hardening 的 hosted 状态在新分支提交推送前保持 **`PENDING_HOSTED`**。
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
21. historical Stage 6 sidecar 必须固定为历史 `corpus:4ff7ba48fd38`、标准仓库路径和整文件 SHA-1 `4ff7ba48fd38a4400e581fac32c43c4217de1ece`，再叠加实算 SHA-256；任意自选 corpus ID、alternate path、自洽但非历史的 hash 或未知字段均 fail closed。
22. 显式 Stage 6 SHA 探针只允许产生 `DIAGNOSTIC_CONTENT_COMPATIBLE` / `DIAGNOSTIC_ONLY`，不能提升为正式 READY；正式路径在 legacy identity 漂移时先于 63 MB 级 JSONL 内容解析失败。
23. reviewed hard-case 的每个 source path 必须被限制在显式 source root 下并现场重算文件 SHA-256；正式运行还需带时区的 release attestation，自报 `source_verification`、路径逃逸、缺文件或漂移都不能取得发布权限。
24. reviewed semantic row 必须绑定 claim/evidence 正文或 SHA-256、拒绝 ID 内容漂移；scorer model 必须是 Hugging Face Hub repo ID，revision 必须为小写完整 40 位 Git commit SHA，实际离线加载的 tokenizer/model commit metadata 也必须精确匹配；precision-first 阈值除 precision/coverage 外，至少需要 5 个 predicted positives 与 5 个 true positives。
25. `Unnecessary Tool Call Rate` 必须以实际调用事件为分母、非 required tool 的调用事件为分子；空 `required_tools` 时任何调用均为 unnecessary，全局主值使用 call-level micro，同时保留 per-case macro。
26. 每个统一 node trajectory event 必须带 `step/node/action/input_summary/output_summary/status/latency_ms/tokens/error_type`；输入输出摘要只记录状态形状、集合规模与计数，不复制 query、claim 或 answer 正文。

## 3. Phase 完成矩阵

| Phase | 本地实现状态 | 已落地证据 | 外部剩余条件 / 发布限制 |
|---|---|---|---|
| A RunIdentity / strict trace | `COMPLETE_LOCAL` | 15 字段 `RunIdentity`；五文件 EvalBundle；staged validation + atomic replace；比较前身份校验；失败行严格 schema；统一 node event 含 M8 的 step/input/output/tokens 合同且摘要不复制正文；current-dev real-corpus bundle 为 READY | 无本地代码缺口 |
| B 统一 Reasoning Plan | `COMPLETE_LOCAL` | 单跳、多跳、LOOKUP/SEARCH/CALCULATE/COMPARE 共用 build→plan→execute→observe→coverage→synthesize→claims→verify→finalize；malformed decomposition fail closed | 无本地代码缺口 |
| C Calculation + Claim | `COMPLETE_LOCAL` | evidence-pool operand reuse；单证据多值拒绝静默选首值；Decimal 计算；显式 calculation lineage；atomic ClaimRecord；parent DAG；DERIVED 重算并绑定 `CalculationRecord`；planned comparison synthesis 强制校验 step/result/parent/calculation lineage；canonical abstain；ENTAILED-only finalize | 无本地代码缺口 |
| D Structured Fact v2 | `COMPLETE_LOCAL` | 显式 `MetricSpec`；ACTUAL/ADJUSTED/FORECAST 严格区分；scope/date/revision/basis；Q2 单季与累计口径不再自动混同；DuckDB filter pushdown；legacy JSONL 与 read-only DuckDB 加载 | 无本地代码缺口 |
| E 四 profile / reviewed metrics | 框架 `COMPLETE_LOCAL`；当前 run 为 `SYNTHETIC_CONTRACT_ONLY` | 四个 canonical profile；共享 comparison identity；完整 per-case trajectory；同参 report-search 跨计划去重；case 级 executor failure；gold/process/contract 指标分层；Unnecessary Tool Call Rate 使用 call-event micro + per-case macro；reviewed claim/calculation 非零分母发布门禁；source-root 文件 SHA 重算 + release attestation；extra calculation 七维惩罚；evidence mismatch 归因；`EXPECTED_ABSTENTION` 与 observed process failure 分离 | 缺正式 reviewed cases/manifest、可核验 source files 与 release attestation；当前 run `performance_claim_allowed=false`，不得发布排名或质量提升 |
| F semantic / 历史资产 | 代码 `COMPLETE_LOCAL`；资产 `BLOCKED_EXTERNAL_ASSETS` | reviewed label schema + claim/evidence 内容哈希；precision-first calibration + 最小 predicted-positive/TP；Hub repo ID-only、local-files-only directional-NLI scorer；小写完整 commit SHA + tokenizer/model 实载 commit metadata 精确绑定；READY readiness + actual labels/hash + calibration + scorer identity 四方绑定；deterministic→NLI→bounded LLM judge；full seed/results 已通过 attestation；historical runtime 前置 Stage 6 门禁；Stage 6 固定 historical corpus ID/path/SHA-1 + current SHA-256 并隔离 diagnostic status | 缺 reviewed semantic labels 及可信 SHA；缺授权且本地缓存完整的 directional-NLI Hub snapshot/config；缺历史 SHA-1 精确匹配、内容兼容的 Stage 6 corpus 与 reviewed attestation；正式 semantic activation 未运行 |
| G CI / docs / observability | 当前 hardening `COMPLETE_LOCAL / PENDING_HOSTED`；上一基线 `HOSTED_GREEN` | `requirements.lock`；固定 `ruff==0.12.12`；11 个 Phase G/CI/评测核心文件的增量 lint + format-check；Ubuntu/Windows CPU-only workflow；provider-free 四 profile step；持久化 verdict；strict HTML；本轮本地 1084 tests、Phase 0 215 tests 与 CI 等价门禁已通过 | 上一基线 run `33260006502`：Ubuntu 2m01s、Windows 3m32s，两个矩阵作业成功；本轮提交后必须等待新的双平台 run 才能恢复当前 HEAD 的 `HOSTED_GREEN` |
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
| CI、全量测试、hard-case subset 和文档验收全通过 | 当前 hardening `COMPLETE_LOCAL / PENDING_HOSTED`；上一基线 `HOSTED_GREEN` | 本轮全仓、Phase 0、十类 balanced subset、编译、Ruff、依赖和 diff 门禁均通过；上一基线 run `33260006502` 在 Ubuntu/Windows 均成功，新 HEAD 仍需新 run |

结论：工程实现已本地闭环，hosted 证据在推送本轮提交后重新验收；由于上表仍有 reviewed/private 资产必要条件为 `BLOCKED_EXTERNAL_ASSETS`，整轮升级必须保持非全局 COMPLETE。

## 4. 当前权威验收结果

### 4.1 全仓测试与静态门禁

- 当前待提交分支：`codex/financial-agent-strict-plan`。本轮提交仍明确排除带人类签署身份且发布授权未单独确认的 untracked `historical-full-raw.attestation.json`。
- 当前工作区完整 suite：`1084 passed, 1 skipped`（29.72s）。唯一 skip 是缺失真实 Stage 6 snapshot 时必须保留的 private restore gate，不能用兼容重建或 synthetic corpus 消除。
- Phase 0 deterministic smoke：`215 passed`。strict observability：`READY`；balanced hard-case subset：PASS；four-profile contract：`SYNTHETIC_CONTRACT_ONLY`。
- 上一冻结基线 hosted deterministic-ci：run `33260006502`，head SHA `446a316103d2c252d1582f0c2091ce23e32efe5e`，Ubuntu `success`（2m01s），Windows `success`（3m32s）。该 run 不覆盖本轮尚未推送的 hardening。
- Balanced hard-case subset：10 cases，10 类各 1 条，subset SHA-256 `5ef1c2dc33de427aeba77a9bf5d564b930dea9e0e5eb07199821f836abfc5dd4`；当前重跑 PASS。
- `uv --no-cache pip check --python .venv\Scripts\python.exe`：通过，`77 packages compatible`。使用 `--no-cache` 是因当前 sandbox 无权读写用户级 `uv` cache，与依赖一致性无关。
- `ruff check`：通过；`ruff format --check`：`11 files already formatted`。该门禁是明确列出的 Phase G/CI/评测核心文件增量作用域，不代表全仓历史样式债已清零。
- `python -m compileall -q src scripts` 与定向 `py_compile`：通过。
- current-dev、strict-observability 和 profile 共享 canonical source manifest：234 files，`hash_policy=text-lf-normalized-v1`，hash `d7382d6e8f6afc39a58857308b6be772ab728790de715a34fe119524310e36ca`；manifest 文件 SHA-256 `06472d69c2907e81b9cd89999d7d1e6c6af8f1b7383b1cab8f542c16e821a20c`。
- `git diff --check`：通过，无 whitespace error；仅有 Windows CRLF 提示。
- 上述通过项及 hosted green 证明当前实现与公开 workflow 合同一致；它们仍不替代 reviewed benchmark、semantic calibration 或 historical corpus replay。

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
- bundle reference SHA-256：`2eca366114ba7751b0052cccada31400b9210fab8efd56516e743dea9682b6f3`
- current-dev summary：`outputs/reports/current_dev_trace_summary.json`，SHA-256 `93d1eb0d521c9d62a1725d5e2441205019b9d361c16a221408fd5544426ce2b1`
- run ID：`run-a8bf80ad9a1d99758b75`
- identity hash：`a8bf80ad9a1d99758b75fdd4a145fb653f3b5dc0bab0d7fb131cc7f2eb727c28`
- source manifest hash：`d7382d6e8f6afc39a58857308b6be772ab728790de715a34fe119524310e36ca`
- corpus hash：`0f90e75c99a612d05540a8760b0ae34ba8ebd8e030fe9e881995f97910ce2c69`
- corpus asset manifest hash：`748bf4f60bf24ea02f429cb3c4c9e16de687ccd08e3d9c585f53dd34ad16fabc`
- evidence integrity：`READY`，1 trace、1 claim、1 evidence reference、0 issue、0 error。
- 独立复核报告：`outputs/reports/evidence_integrity_dev_final_20260829.json`，SHA-256 `2809c9db8e717751e20bb351fb4e5d4d4e5d5d8e76481a4a02f3d5aaeafd4098`；其 run ID/identity 与 bundle 精确一致。
- 路径：真实 `facts.duckdb` → 两个 structured operands → `net_margin` → DERIVED claim / `CalculationRecord` → deterministic verification → final citation。
- scope：`REAL_CORPUS_DEV_CONTRACT_ONLY`、`publishable=false`、`quality_claims_allowed=false`。
- 旧 `outputs/reports/agent_traces_dev.jsonl` 未覆盖，继续作为 BLOCKED migration fixture。

该 run 证明真实本地 corpus 上的严格计算与证据链闭环，不是 reviewed benchmark 性能证据。

### 4.3 合成 100-case 四 profile 执行合同

运行输出：`outputs/profile_ablation_contract_final_20260829/profile_ablation_summary.json`。

身份与固定清单：

- status：`SYNTHETIC_CONTRACT_ONLY`
- `performance_claim_allowed=false`
- benchmark version：`contract-v1`
- 100 cases，10 类 × 10；其中 10 个 `must_abstain` case 不产生 claim，所以 graph-backed profile 各为 90 claims。
- cases file SHA-256：`32576e9f9acf39b0905841f221ff22ac2bf1a817fe63468aa8f1a801e72fe29f`
- canonical benchmark hash：`44be5c94e0b9c94aaa2c77bfad10352802fff4d6e4b363c76746182825d36de9`
- case IDs hash：`6690f3f46898e6d9ef17462135438d0f4038aaff36cb63c3346275188980f45c`
- manifest：`benchmarks/hard_cases/contract-v1.manifest.json`，SHA-256 `8cce7a6058189bd5dfd383150f8649d324bc92a98b0721bce2faaf5f07ac0bba`
- summary：`outputs/profile_ablation_contract_final_20260829/profile_ablation_summary.json`，SHA-256 `dd627ea1f8d13c457bf5c2e5cd8335869743ed8d1cb006525e6e3a12f22c3404`
- comparison context hash：`93c92a4b292437a5328cb0dfa411a8e2822bf805ac5c6d121c5ed85bf4c9979a`
- shared source manifest hash：`d7382d6e8f6afc39a58857308b6be772ab728790de715a34fe119524310e36ca`
- source manifest file SHA-256：`06472d69c2907e81b9cd89999d7d1e6c6af8f1b7383b1cab8f542c16e821a20c`
- shared corpus hash：`ff2250cb89e96393efa8f77cb4896faf60a75bcb39ffcc39f3e041330742ba0e`
- corpus asset manifest hash：`a3cf136a64ce23e90f7dbd2eb4eec54a9753d15fdf8b7a02010ed17717547076`
- contract oracle chunks SHA-256：`3f9592fb15e50c35c3beb82ee806b2dd77eddceef0aac82c6bd43e615690dcfe`
- materialized structured facts SHA-256：`9d24f5da0309e3b3b68d0312bba99fa3be584cea345e77b8d12587e796f75b3f`
- materialized company aliases SHA-256：`9d3f103dfa2f930e4980aed458b3bfcf82d403ce0809038dbcde6a8bf5191d7f`
- base identity git commit：`70069a0da31a593de30f3163c6890c15a2551f06`
- provider/model：`offline-deterministic` / `synthetic-contract-evidence-echo`；无外网、无模型下载、gold 不进入 answerer。

| Profile | Run ID | Evidence integrity | 持久化摘要 |
|---|---|---|---|
| baseline-rag | `run-6ee218f419e426996622` | `NOT_APPLICABLE` | baseline 不生成 verified claims；该状态是设计语义，不是缺失 gate |
| agentic-rag | `run-313b04b71af37e88aee3` | `READY` | 100 rows / 90 claims / 90 entailed evidence refs / 0 issues / 0 errors |
| structured-agent | `run-08741640f2623a05887d` | `READY` | 100 rows / 90 claims / 90 entailed evidence refs / 0 issues / 0 errors |
| full-agent | `run-786f38d2f468740f97b4` | `READY` | 100 rows / 90 claims / 90 entailed evidence refs / 0 issues / 0 errors |

四个 identity 已通过可比性校验。每个 profile 的 must-abstain 汇总均为 `failed_query_count=0`、`failure_rate=0.0`、`expected_abstention_count=10`、`observed_process_failure_query_count=10`；这表示正确 abstention 不再被误计为结果失败，同时保留过程诊断。这里的 READY 只表示 graph-backed bundle 的证据引用与持久化工件一致；baseline 的 `NOT_APPLICABLE` 和三个 graph profile 的 READY 都不能把 synthetic contract 变成 reviewed 性能结果。

需要特别区分“矩阵完整执行”和“treatment 被题型触发”：当前
`derived_calculation` 与 `multi_hop` synthetic rows 通过 `SEARCH` 完成且没有
persisted calculation，因此 100-case 结果不能证明 calculation/multi-hop treatment
activation。该行为由 `tests/test_profile_runtime.py` 单独验证；正式效果仍只能由
reviewed benchmark 判定。

### 4.4 Synthetic strict observability contract

- summary：`outputs/strict_observability_smoke_final_20260829/strict_smoke_summary.json`，SHA-256 `95d6c1cfd4020c362980e0ad17620880164bfdc0f21b0b28c9b1400c7d58c202`
- bundle reference：`outputs/strict_observability_smoke_final_20260829/strict_smoke_bundle.json`
- run ID：`run-773279e2cebc8e8fb02c`
- identity hash：`773279e2cebc8e8fb02c3243b44c0f7a97ddf28565015f19778ef6a4003bbaa4`
- canonical source manifest hash：`d7382d6e8f6afc39a58857308b6be772ab728790de715a34fe119524310e36ca`，与 current-dev/profile 完全一致。
- persisted integrity：`READY`，1 trace / 1 claim / 2 entailed evidence refs / 0 issues / 0 errors。
- 结果：真实生产图、YoY `CalculationRecord`、ENTAILED claim、持久化 READY verdict 与 HTML 均通过。
- scope：`SYNTHETIC_CONTRACT_ONLY`、`publishable=false`、`quality_claims_allowed=false`。

### 4.5 Historical full 原件与门禁

2026-08-24 授权恢复后，historical full 的两个原件已按原字节恢复并由独立 attestation 绑定；没有给历史 JSONL 补造逐行 metadata：

- full seed：`data/eval_set/answer_eval_seed_full.jsonl`，50 行，SHA-256 `ecd8fd20cd6beff9a6e4ba9df9b795e97cca1c96a4705d82952489d61c132d21`，状态 READY。
- historical results：`artifacts/remote_20260401/outputs_reports/answer_eval_results_full.jsonl`，50 行，SHA-256 `4940210e2dd6ac8309852581304eb65ad12d445e59b2ae1b892d94c01f35900e`，状态 READY。
- restoration report SHA-256：`fb5235ab42283438451133899eb64b82087db7b53c08a298d0d3b22cea53ab47`。
- attestation：`benchmarks/full/historical-full-raw.attestation.json`，SHA-256 `1ad436865eb25824e3be2c1f6674c98b946df67f0602060b56c9ecc6d9bc97ee`；`check-full-seed` 为 READY。
- 最新 readiness：`outputs/phase0/readiness_authorized_assets_final_20260829.json`，SHA-256 `153029eaf5c3ff8f605edfa69eba25b0b1f5e2181aaeff26890671056b4a53cb`；其中 full seed/results 为 READY，顶层仍因 semantic labels 缺失而 BLOCKED（预期退出码 2）。

50-row seed 共涉及 54 个唯一 gold chunk；历史 results 为其中 42 个保留了同 ID 的 snippet/page/file 锚点，另外 12 个只有 reviewed badcase gold 与 attestation 背书。单文档兼容探针虽然复现了历史 doc ID 与 `c0004`/`c0006`，但文本与历史锚点错位，因此：

`historical-full-raw.attestation.json` 当前仅作为本地 reviewed gate 输入；它包含
具体 `attested_by` 人类身份，未获明确签署/发布授权前不得自动 stage 或提交。
其本地 READY 结论不能被解释为该人类 attestation 已获公开发布授权。

- 该 78-chunk 探针只能标记为 `COMPATIBILITY_REBUILD_AUDIT_ONLY`。
- 冻结的兼容重建审计 `outputs/phase0/historical_full_evidence_probe_20260824.json` 状态为 BLOCKED，SHA-256 `8f9e921c1b5ac13019feb78539f93fc2f297efee4ed2de25b9d73d99d065811a`。
- 最新外部门禁 `outputs/phase0/historical_full_evidence_probe_final_20260829.json` 状态仍为 BLOCKED，SHA-256 `44f5c20a69c1206e4d93d7fb892a0904d50d96f3b84d3f777a7fc3fc13cfd930`（预期退出码 2）；当前明确缺少 Stage 6 chunks snapshot 与 reviewed corpus sidecar。
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
| Historical Stage 6 replay | 是 | `corpus:4ff7ba48fd38`；整文件 SHA-1 `4ff7ba48fd38a4400e581fac32c43c4217de1ece`；内容、页码、文档身份均与 historical gold 对应的 hash-pinned `tmp_stage6_data/chunks/chunks.jsonl` | 当前 23942-chunk corpus 与 ID-compatible rebuild 均已证明不能替代历史 snapshot |
| Stage 6 provenance | 是 | `benchmarks/full/historical-stage6-corpus.attestation.json`，需由资产负责人绑定来源、owner、review、chunk count 与 seed/results hashes | observed SHA 只能证明看到某文件，不能证明其来源与 reviewed 身份 |
| Reviewed 四 profile 性能比较 | 是 | `benchmarks/hard_cases/reviewed_cases.jsonl` 与 `reviewed_manifest.json`，以及正式模型的 pinned revision | synthetic 100 只验证执行合同，不能生成可发布的 reviewed Accuracy 或 profile 排名 |

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
  --reviewed-source-root <reviewed-source-root> `
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
- hosted CI 在 Ubuntu 与 Windows 均实际成功（已由 run `33259417431` 满足）。
- README 或报告中的任何性能数字都直接引用 reviewed bundle，而不是 synthetic/current-dev contract。

在这些条件全部满足前，项目总状态保持 `BLOCKED_EXTERNAL_ASSETS`，不得写成全局 COMPLETE，也不得宣称 reviewed 性能提升。

两个不阻塞本地合同、但仍值得后续硬化的项目是：GitHub Actions action tags 与 lock 中 wheels 尚未做不可变 SHA/hash pin；`offline` 环境变量只证明当前被测路径无需 provider 或模型下载，不是操作系统级网络防火墙。
