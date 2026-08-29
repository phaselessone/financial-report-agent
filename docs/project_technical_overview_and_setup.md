# 金融研报 RAG 项目技术说明与搭建说明

## 0. 当前验收状态（2026-08-29）

这份文档同时保留“当前实现说明”和“2026 年 4 月历史实验快照”。两类证据
不可混用：历史模型、远端路径、指标和 `73 passed` 只说明当时的状态，不是
当前版本的验收结论。

- Phase 0 readiness: `BLOCKED`。经人工复核并固定 SHA-256 的 50-row full seed
  与历史 full results 已恢复并通过 attestation；reviewed semantic labels 仍缺失。
  历史 full 重放还单独缺少 hash-pinned、内容兼容的 Stage 6 chunks snapshot，
  因此不能宣称 full benchmark、semantic calibration 或四 profile 效果已通过。
- 当前可验证的是离线 deterministic contract：统一 `ReasoningPlan` 图、结构化
  lookup/search/calculation、claim/citation/calculation provenance、严格
  `RunIdentity` 与 strict eval bundle。
- `scripts/strict_observability_smoke.py` 会跑真实图节点并写入持久化、与 bundle
  hash 绑定的 `READY` integrity verdict；该工件永久标记
  `SYNTHETIC_CONTRACT_ONLY`、`publishable=false`，不能作为质量提升证据。
- `scripts/regenerate_dev_trace.py` 从当前 `chunks.jsonl`、`facts.duckdb` 和
  `company_aliases.json` 选择一个无歧义真实语料案例，跑结构化查询、确定性
  `net_margin`、claim verification 和 strict bundle，生成
  `outputs/reports/agent_eval_bundle_dev.json`。该工件标记
  `REAL_CORPUS_DEV_CONTRACT_ONLY`、`publishable=false`；旧
  `agent_traces_dev.jsonl` 保留为 BLOCKED migration fixture。
- `.github/workflows/ci.yml` 是 Ubuntu/Windows CPU-only 的可复现工作流合同；
  目前没有可作为终验依据的 hosted GitHub Actions 成功记录。
- 一个不含 `data/` 和 human attestation 的 263-file clean-checkout 本地快照已通过
  compile、Ruff、`uv pip check`、59 项 RunIdentity/eval 回归、31 项
  evidence/Phase-G 回归、1009 项 public suite、strict READY、balanced hard-case
  subset、四 profile synthetic contract 和 Phase 0 的 200 项 smoke；这仍是本地
  模拟，不是 hosted CI 成功记录。
- `make four-profile-contract` 会在无 provider、无网络、无模型下载的条件下真实执行
  100 条 synthetic contract × 四个 profile，生成四个独立 bundle。该结果永久为
  `SYNTHETIC_CONTRACT_ONLY`；三条 graph-backed profile 必须通过 evidence-integrity，
  baseline 因不含 verified-claim 层而明确标为 `NOT_APPLICABLE`。
- 100-case 四 profile 只验证入口、身份、bundle 和 integrity 合同，不证明每一类别
  都激活同名 treatment；当前 `derived_calculation` / `multi_hop` rows 仍通过
  `SEARCH` 完成且 calculation count 为 0，具体 treatment activation 由
  `tests/test_profile_runtime.py` 独立覆盖。
- 正式 semantic 激活只接受 reviewed readiness/calibration JSON 加一个匹配
  `benchmarks/semantic/directional-nli-config.schema.json` 的本地 scorer config；配置
  hash、labels hash、calibration report hash 和模型 revision 全部进入 RunIdentity，且
  强制 `local_files_only=true`、`trust_remote_code=false`。
- 当前复现与验收命令以本文第 8 节和 README 为准；第 6、9 节明确属于
  **历史快照（非当前验收）**。

## 1. 项目定位

这个项目不是一个通用聊天机器人，而是一套面向金融/行业研报 PDF 的端到端 RAG 系统。目标是把大量结构复杂、更新频繁、带表格和标题层级的研究报告，转成可检索、可引用、可评测的问答系统。

当前系统覆盖六条主线：

1. PDF 解析、清洗、分块与中间产物落盘
2. Dense / BM25 / Hybrid / Rerank 检索链路
3. 基于证据的答案生成、引用绑定、支持性校验与拒答
4. 检索级与答案级评测、坏例收集、运行归档
5. LangGraph 状态图上的 `ReasoningPlan -> execute_step -> dependency_gate`
6. claim / citation / calculation provenance 与 strict eval bundle 完整性门禁

代码主入口位于：

- `run_pipeline.py`：PDF ingest 与 chunk 产物生成
- `run_prepare_benchmark.py`：评测物料准备
- `run_retrieval_eval.py`：检索评测
- `run_answer_eval.py`：答案生成与答案级评测
- `run_agent.py` / `run_agent_eval.py`：Agent 图执行与 strict bundle 评测
- `run_profile_ablation.py`：四 profile 合同执行；正式比较必须使用外部 reviewed 资产

## 2. 为什么做 RAG，而不是直接微调

参考你给的三张图，这个项目的核心叙事可以这样讲：

1. 研报 PDF 更新快、行业跨度大，知识具有明显时效性，直接把知识塞进模型参数并不经济。
2. 问题很多都要求给出具体报告、具体页面、具体数字，单纯生成答案不够，必须能追溯到证据。
3. 金融 PDF 噪声重，存在目录、免责声明、表格、页眉页脚、网页导出的杂质，通用问答模型很难直接处理，需要单独的数据工程和检索工程。
4. 项目更强调“可验证、可复现、可评测”，所以 RAG 比纯微调更适合作为第一阶段落地方案。

一句话概括：这是一个“面向金融研报 PDF 的证据驱动型 RAG 系统”，不是“把大模型接个向量库”的简单 demo。

## 3. 业务问题与支持的问题类型

系统当前支持四类典型问题：

1. `numeric_fact`：数字事实题，例如价格、同比、增速、规模、出货量
2. `report_lookup`：报告定位题，例如“哪份报告提到了 X”
3. `comparison`：比较题，例如“几份报告分别怎么判断”
4. `inductive`：归纳题，例如“近期报告共同强调了什么”

这四类问题与研报阅读场景高度贴合，也决定了系统不能只做单路召回或单阶段生成。

## 4. 整体架构

### 4.1 数据流

当前主数据流为：

`pdf/` -> `data/parsed_pages` -> `data/cleaned_pages` -> `data/elements` ->
`data/chunks` / structured facts -> `ReasoningPlan` -> `execute_step` ->
`dependency_gate` -> synthesize -> claim verification -> strict eval bundle

以下 stage6 路径仅是历史压力测试快照，不是当前验收默认路径：

`data/raw_pdfs/incoming` -> `tmp_stage6_data/chunks/chunks.jsonl` -> `outputs/` / `artifacts/`

### 4.2 模块分层

#### A. Ingest 层

对应目录：`src/ingest/`

职责：

1. `parser.py`：解析 PDF 页面与基础文本结构
2. `cleaner.py`：去掉页眉页脚、空白页、网页导出噪声
3. `chunker.py`：构造多种 chunk 策略并输出父子块
4. `pipeline.py`：串起解析、清洗、切块、报告统计与坏例记录

当前保留了三套 chunk 策略实验产物：

- `fixed_window`
- `title_aware`
- `table_protected`

生产策略最终固定为 `table_protected`。原因很直接：金融 PDF 里大量问题与标题边界、表格结构、数字邻近上下文有关，窗口切太大容易把噪声和正文揉在一起，切太粗又不利于证据定位。

#### B. Retrieval 层

对应目录：`src/retrieval/`

职责：

1. `embedder.py`：Dense embedding 构建与缓存
2. `faiss_index.py`：FAISS 向量索引
3. `bm25_index.py`：BM25 关键词检索
4. `hybrid_retriever.py`：Dense + BM25 融合
5. `reranker.py`：Cross-Encoder 二次排序
6. `runtime.py`：统一检索入口与 query 特征处理

当前默认链路不是单路向量检索，而是：

`Dense + BM25 -> Hybrid -> Rerank -> TopK Evidence`

其中 `runtime.py` 还做了几件很关键的工程动作：

1. 根据 query 判断是否为数字题、比较题、归纳题
2. 对 BM25 query 做扩展，提升术语匹配能力
3. 过滤明显的目录/免责声明/报告汇总类噪声 chunk
4. 把 child chunk 扩展为带 parent context 的证据行

#### C. Generation 层

对应目录：`src/generation/`

职责：

1. `provider.py`：本地模型与 DeepSeek provider 切换
2. `prompt.py`：不同题型的 prompt 模板
3. `answerer.py`：证据选择、分题型生成、JSON 解析、fallback、引用绑定
4. `citation_builder.py`：引用构建
5. `abstain.py`：支持性校验、置信度和拒答决策

生成层不是“把 top-k chunk 拼进 prompt 里就结束”。

当前实现里：

1. 比较题和归纳题采用两阶段生成，不是一轮 prompt 生硬合成
2. 最终答案必须通过支持性校验，避免“看起来像对，实际上证据不够”
3. 生成结果会绑定引用，尽量把答案落到具体证据文档
4. 当证据不足、文档守卫触发或归纳不可靠时，会走 fallback 或 abstain

#### D. Evaluation 层

对应目录：`src/evaluation/`

职责：

1. `retrieval_eval.py`：检索召回与排序评测
2. `answer_eval.py`：答案语义、引用、支持性、拒答与延迟评测
3. `benchmark_assets.py`：评测集物料构建
4. `output_archive.py`：产物归档
5. `run_metadata.py`：运行元数据与 source manifest

这一层是项目工程价值很高的地方。系统不是只给出一个答案，而是能解释：

1. 这次跑的是哪套语料
2. 这次跑的是哪套 benchmark profile
3. 哪些题错了
4. 错在哪种题型
5. 哪些是召回问题，哪些是生成问题，哪些是引用问题

#### E. Agent 与结构化事实层

对应目录：`src/agent/`、`src/structured/`

当前 LangGraph 主链路为：

```text
analyze_query
  -> build_reasoning_plan
  -> plan_next_step -> execute_step -> observe_step_result -> repeat
  -> dependency_gate
  -> grade_evidence / rewrite_query（仅在需要时）
  -> synthesize -> extract_claims -> verify_answer -> finalize
```

所有问题都进入同一个 `ReasoningPlan`。`execute_step` 统一执行 `LOOKUP`、
`SEARCH` 和 `CALCULATE`，并保留完整 period/value-type/accounting-scope/revision
坐标、依赖、输入 evidence/fact、公式和舍入记录。最终非拒答内容必须经过原子
claim 提取与确定性 provenance 校验；只有 `ENTAILED` claim 的证据可以进入引用。

## 5. 技术选型与理由

## 5.1 当前实际采用的技术

| 层级 | 方案 |
|---|---|
| PDF 解析 | PyMuPDF |
| Dense Embedding | `BAAI/bge-m3` |
| Dense Index | FAISS |
| Sparse Retrieval | BM25 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Agent orchestration | LangGraph 统一状态图 |
| Structured facts | DuckDB + metric/period/value-type registry |
| LLM Provider | 可选 DeepSeek/OpenAI-compatible API 或本地模型；非 deterministic 验收前提 |
| 评测输出 | JSON / JSONL / Markdown |
| 远端运维 | `scripts/remote_ops.py` + Paramiko |

## 5.2 与图片中的常见方案关系

你给的图片里提到的 RAGFlow、FAISS、FlagEmbedding、LangChain-ChatChat、QAnything、LlamaIndex、Milvus、GraphRAG，更适合放在“参考路线”里介绍，而不是直接说当前项目用了全部这些组件。

更准确的说法是：

1. 当前项目借鉴了成熟 RAG 工程里的思路，但实现上偏“轻量自建链路”
2. 当前核心依赖包括 PyMuPDF、FAISS、BGE embedding/reranker、LangGraph、DuckDB；DeepSeek/Qwen 是可选生成 provider
3. 当前已使用 LangGraph 编排，但未采用 LangChain-ChatChat、Milvus、Elasticsearch 或 GraphRAG；是否引入它们取决于规模和检索需求，而不是验收口号

如果在面试里被问“为什么不用 Milvus / LangChain / GraphRAG”，可以直接答：

1. 当前语料规模还没逼到必须上分布式向量库
2. 项目优先解决 PDF 结构化、混合检索和可追溯评测
3. 先把单机版本闭环做实，比堆中间件更有价值

## 6. 历史快照（非当前验收）

本节归档 2026 年 4 月旧远端部署、旧 provider 配置、旧路径与旧指标，便于
追溯演进。它们没有绑定当前 strict `RunIdentity`、`requirements.lock`、reviewed
资产和 READY 门禁，**不得作为当前版本验收、profile 排名或质量提升声明**。

### 6.1 环境与测试

1. 历史记录称远端健康检查、DeepSeek 配置和当时评测种子可用
2. 历史记录的测试结果为 `73 passed`；该数字不是当前测试总数或终验结果
3. 历史远端部署路径为 `/root/autodl-tmp/financial-rag-project`
4. 历史实例曾配置 `.env.deepseek` / `.env.runtime`

### 6.2 数据与 chunk

历史文档记录的基线规模为：

- PDF：54 份
- 页数：680
- 生产 chunk：4239
- ingest badcase：10

这些数字来自项目复盘文档，说明 ingest 端已经不是空壳，而是有实打实的语料和坏例管理。

### 6.3 检索结果

历史远端 `outputs/reports/retrieval_eval_summary.json` 曾记录以下 48 题结果：

| 方法 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| Dense | 0.5417 | 0.6458 | 0.3763 |
| BM25 | 0.5000 | 0.5833 | 0.3666 |
| Hybrid | 0.5417 | 0.7083 | 0.4000 |
| Hybrid+Rerank | 0.6667 | 0.6667 | 0.6038 |

该 2026 年 4 月历史快照中，`Hybrid+Rerank` 的上述指标最好；这不能证明它仍是
当前版本最强或当前生产环境的最优入口。

### 6.4 答案评测结果

#### Dev 集（current-dev，16 题）

来自 `artifacts/remote_20260410_dev_repro_v6/reports/answer_eval_summary_dev.json`

| 指标 | 数值 |
|---|---:|
| Answer Hit Rate | 0.7500 |
| Citation Hit Rate | 0.6250 |
| Citation Span Hit Rate | 0.5000 |
| Support Hit Rate | 1.0000 |
| Fallback Rate | 0.1875 |
| Failed Query Count | 0 |

该 2026 年 4 月 dev 快照只说明当时已进入可继续迭代的状态，不能作为当前版本验收。

#### Full Core 历史压力集（historical-full-core，40 题）

来自 `artifacts/remote_20260410_full_core_repro_v5/reports/answer_eval_summary_full.json`

| 指标 | 数值 |
|---|---:|
| Answer Hit Rate | 0.6857 |
| Citation Hit Rate | 0.8286 |
| Citation Span Hit Rate | 0.4000 |
| Support Hit Rate | 0.9143 |
| Abstain Precision | 0.8333 |
| Abstain Recall | 1.0000 |
| Unsupported Answer Count | 3 |
| Fallback Rate | 0.1500 |
| Failed Query Count | 0 |

在该历史 core 压力集中，系统能够完整执行；当时观察到的主要短板是
span-level citation 与复杂 comparison/inductive 表达，拒答策略也开始产生作用。
这些都是历史诊断，不是当前 reviewed benchmark 结论。

### 6.5 一个必须讲清楚的 benchmark 口径

这里有一个很重要的口径问题，面试、汇报和文档里都要说清楚：

1. 项目里恢复过一个 `full 50` 历史压力集，这是 provenance 文档里明确写过的
2. 但当前默认的 `answer-eval-full` 实际跑的是 `historical-full-core`，对应 40 条核心题，而不是 50 条全量压力题
3. 所以如果你对外说“full 50 当前指标是多少”，必须先确认你说的是 `full raw 50` 还是 `full core 40`

这个点不说清楚，会让外部读者误以为所有“full”指标都能直接横向对比。

## 7. 当前主要工程风险

结合代码、文档、远端产物和坏例，可以把当前风险归纳为四类：

### 7.1 复杂题型泛化仍然不稳

`comparison` 和 `inductive` 题仍然是主要难点。原因不是召回完全失败，而是：

1. 多篇相似报告需要更稳定的排序与证据聚合
2. 归纳题容易在“抽象总结”和“证据严格对齐”之间失衡
3. doc 命中和 span 命中之间仍有明显落差

### 7.2 文档口径存在阶段漂移

早期复盘文档记录的是 4 月 5 日左右的结果，和 4 月 10 日后的最新复现实验已经不一致。对内问题不大，但如果直接拿旧文档对外讲，会低估或误讲当前状态。

### 7.3 测试入口与证据边界

项目标准回归入口是 `python -m pytest -q`。旧记录中“该命令在远端通过、裸
`pytest` 收集失败”只描述 2026 年 4 月环境。当前可确认的是本地 clean-checkout
simulation 已通过 1009 项 public suite；尚无当前提交对应的 hosted
Ubuntu/Windows CI green。

### 7.4 生成延迟仍是主要耗时项

检索延迟通常在毫秒级，但完整答案生成仍明显慢于检索，后续如果要进一步产品化，需要继续控制：

1. 上下文长度
2. comparison / inductive 多阶段调用成本
3. fallback 与 abstain 带来的额外校验成本

## 8. 本地搭建说明

## 8.1 环境要求

验收环境：

1. Python 3.12
2. Ubuntu 或 Windows（CI 合同的两个目标；hosted 成功状态仍待外部证明）
3. deterministic gate 使用 CPU-only PyTorch，不要求 GPU、模型下载或 API key
4. 真实 embedding/reranker/生成运行可另外配置 GPU 和 provider

可复现安装使用 `requirements.lock`。`requirements.txt` 与
`requirements-dev.txt` 是更新 lock 时的开发输入；不能用未锁定安装替代终验。
核心依赖包括：

- PyMuPDF
- numpy
- faiss-cpu
- jieba
- rank_bm25
- sentence-transformers
- transformers
- modelscope
- torch
- httpx
- LangGraph
- DuckDB

## 8.2 创建虚拟环境

Linux / WSL:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock
```

Windows PowerShell:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
```

如果在开发环境中需要调整依赖或运行远端运维脚本，分别查看
`requirements-dev.txt` / `requirements-ops.txt`，更新后应重新生成并审核 lock：

```bash
pip install -r requirements-ops.txt
```

## 8.3 环境变量

项目通过 `scripts/use_env.sh` 按优先级加载：

1. `.env.deepseek`
2. `.env.local`
3. `.env.runtime`

可以先参考 `.env.example` 创建配置，例如：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=60
DEEPSEEK_MAX_RETRIES=3
REMOTE_HOST=connect.westb.seetacloud.com
REMOTE_PORT=25485
REMOTE_USER=root
REMOTE_KEY_PATH=.ssh/financial_rag_ed25519
```

如果走 DeepSeek，需要再补充：

```env
DEEPSEEK_API_KEY=你的密钥
```

## 8.4 准备语料

把当前语料放到：

```text
pdf/
```

如果要重放历史 full core/raw 快照，需要恢复并校验 stage6 历史语料；以下是
legacy 路径，不是可直接声明 READY 的当前资产：

```text
tmp_stage6_data/chunks/chunks.jsonl
data/eval_set/answer_eval_seed_full.jsonl
```

候选 Stage 6 不能只靠 gold chunk ID 存在来验收。必须同时取得资产负责人提供的
reviewed sidecar（合同见
`benchmarks/full/historical-stage6-corpus.attestation.schema.json`），再运行内容级门禁：

```bash
python scripts/historical_full_evidence_gate.py \
  --chunks-path tmp_stage6_data/chunks/chunks.jsonl
```

该门禁同时核对 corpus 来源/owner/review、hash、文本、页码、doc ID 和文件名；
任何 ID-only 的兼容重建都不能替代历史 snapshot。显式 SHA 只用于诊断探针，不能
替代 reviewed corpus attestation。

## 8.5 健康检查

推荐先跑：

```bash
bash bootstrap/check_env.sh
```

或者：

```bash
bash scripts/ragctl.sh doctor
```

## 8.6 运行链路

### A. 生成 ingest / chunk

```bash
python run_pipeline.py --input-dir pdf --output-dir data --badcase-dir outputs/badcases
```

或：

```bash
bash scripts/ragctl.sh pipeline
```

### B. 准备 benchmark

```bash
python run_prepare_benchmark.py
```

或：

```bash
bash scripts/ragctl.sh prepare-benchmark
```

### C. 跑检索评测

```bash
python run_retrieval_eval.py --prepare-benchmark
```

或：

```bash
bash scripts/ragctl.sh retrieval-eval
```

### D. 跑当前语料 dev 答案评测

```bash
python run_answer_eval.py --split dev --benchmark-profile current-dev
```

或：

```bash
bash scripts/ragctl.sh answer-eval-dev
```

### E. 跑历史 core 压力集

```bash
python run_answer_eval.py \
  --split full \
  --benchmark-profile historical-full-core \
  --model-revision <pinned-provider-revision> \
  --chunks-path tmp_stage6_data/chunks/chunks.jsonl \
  --data-dir tmp_stage6_data \
  --corpus-label stage6-historical \
  --answer-seed-full-path data/eval_set/answer_eval_seed_full.jsonl
```

或：

```bash
bash scripts/ragctl.sh answer-eval-full --model-revision <pinned-provider-revision>
```

历史 full 评测采用不可覆盖的 profile/run 目录：

```text
outputs/eval_profiles/historical-full-core/runs/<profile_run_id>/
outputs/eval_profiles/historical-full-raw/runs/<profile_run_id>/
```

每个 run 根目录独立保存 `benchmark/`、`reports/`、`badcases/` 和
`RUN_POLICY.json`；后续运行不会覆盖既有 run。结果 JSONL 的每一行与 summary
都会携带同一个完整 `RunIdentity`。profile-aware reader/merge 会拒绝 core 与
raw 混用；旧参数 `historical-full` 只作为 `historical-full-raw` 的兼容别名。
`current-dev` 继续使用原有 `outputs/reports`、`outputs/badcases` 布局，兼容现有
脚本与文件路径。

### F. 跑测试

当前公共 deterministic 回归（不伪造缺失 private full seed）：

```bash
python -m pytest -q --ignore=tests/test_full_seed_restore.py
```

终验门禁：

```bash
python scripts/phase0_gate.py readiness
python scripts/historical_full_evidence_gate.py
python scripts/strict_observability_smoke.py --output-root outputs/strict_observability_smoke
```

当前本地 full seed/results/attestation 已通过身份门禁；顶层 `readiness` 仍因
reviewed semantic labels 缺失而返回 `BLOCKED`。只有
`write_validated_eval_bundle` 写出的 agentic bundle 才持久化 bundle-bound READY
verdict；observability 临时重算通过不能把无 verdict 的 bundle 冒充为 READY。
历史 evidence gate 是独立门禁，必须同时取得内容兼容、hash-pinned 的 Stage 6
chunks 和 reviewed corpus attestation。显式 `STAGE6_CHUNKS_SHA256` 只能固定诊断
探针，不能让缺少 reviewed sidecar 的正式门禁变为 READY。

## 9. 历史远端部署快照（非当前验收）

以下信息来自旧审查记录，仅用于定位历史工件：

- 主机：`connect.westb.seetacloud.com:25485`
- 用户：`root`
- 部署目录：`/root/autodl-tmp/financial-rag-project`
- 虚拟环境：`/root/autodl-tmp/financial-rag-project/.venv`

历史记录曾写明：

1. `bash bootstrap/check_env.sh` 通过
2. 当时 `python -m pytest -q` 为 `73 passed`；这不是当前终验测试计数
3. 远端保留了从 `tmp_stage3_*` 到 `tmp_stage15_*` 的多轮实验目录
4. 远端 `artifacts/` 中保留了 4 月 10 日的多轮 dev / full repro 产物

这些记录说明项目曾有远端实验，但由于没有当前 `RunIdentity`、lock hash、reviewed
资产与 hosted CI 证明，不能据此推断当前版本 READY。

## 10. 面试/答辩时怎么介绍这个项目

可以按下面这条主线讲：

1. 业务问题：金融研报 PDF 结构复杂、更新快、必须可引用，所以单纯微调不合适，先做 RAG
2. 数据工程：我不是直接把 PDF 丢给模型，而是自己做了解析、清洗、标题感知和表格保护 chunk
3. 检索工程：不是单纯向量检索，而是 Dense + BM25 的混合召回，再用 reranker 做最终排序
4. 生成工程：对 comparison / inductive 做了分题型处理，并且加了证据绑定、支持性校验和拒答
5. 评测工程：不仅有 Recall/MRR，还有 answer hit、citation hit、support hit、abstain 和 badcase 归档
6. 工程结论：current-dev、strict observability 与 synthetic four-profile 已证明当前执行链和证据合同可运行；reviewed benchmark 性能、semantic activation 和 historical Stage 6 replay 仍被外部资产阻塞。第 6 节性能数字只能作为 2026 年 4 月历史快照介绍

如果对方继续追问，优先准备下面这些点：

1. 为什么 chunk 不是越大越好
2. 为什么不能只用向量检索
3. reranker 到底带来了多少收益
4. 为什么 comparison / inductive 比事实题难
5. 为什么要做 abstain，而不是强行回答
6. 为什么“full core 40”与“full raw 50”要分开讲

## 11. 后续建议

如果你准备继续把这个项目往产品化或简历项目深化，我建议下一步优先做三件事：

1. 把 `full core`、`full raw`、`current dev` 三套 benchmark 名称和命令入口彻底讲清楚，避免口径混乱
2. 继续攻 comparison / inductive 的 span 级引用命中率
3. 把当前这份文档进一步收敛成 README 版本和面试版两个口径
