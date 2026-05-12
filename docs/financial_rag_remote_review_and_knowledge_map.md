# 金融研报 RAG 项目远程审查与知识地图

生成日期：`2026-04-13`  
审查对象：
- 本地设计真相：`F:\code\financial——agent`
- 远端运行真相：`/root/autodl-tmp/financial-rag-project`

## 结论摘要

- [源码事实] 这个项目不是“把大模型接到向量库上”的演示仓库，而是一套面向金融研报 PDF 的证据驱动 RAG 闭环，主链路完整覆盖 `PDF ingest -> chunk -> retrieval -> generation -> citation -> abstain -> evaluation -> artifact`。
- [远端运行事实 | 2026-04-13] 远端 `/root/autodl-tmp/financial-rag-project` 上使用命令 `.venv/bin/python -m pytest -q`，结果为 `73 passed`，说明当前仓库不是概念稿，而是已通过回归测试并具备可运行闭环。
- [远端运行事实 | 2026-04-13] 当前远端 `outputs/reports/retrieval_eval_summary.json` 对 48 道检索题给出 `Hybrid+Rerank` 的 `Recall@5 = 0.6667`、`MRR@10 = 0.6038`；`outputs/reports/answer_eval_summary_dev.json` 对 16 道 dev 题给出 `Answer Hit Rate = 0.75`；`outputs/reports/answer_eval_summary_full.json` 对 40 道 `historical-full-core` 题给出 `Answer Hit Rate = 0.6857`。
- [远端运行事实 | 2026-04-13] 远端归档 `artifacts/remote_20260410_full_raw_repro_v2/reports/answer_eval_summary_full.json` 对 50 道 `historical-full-raw` 题给出 `Answer Hit Rate = 0.5778`、`Citation Doc Hit Rate = 0.6444`、`Support Hit Rate = 0.9333`。这说明 full-core 和 full-raw 不是同一个 benchmark，绝对不能混讲。
- [基于源码的推断] 项目真正的技术含量在四个地方：结构化 PDF ingest、混合检索与 rerank、证据约束生成与拒答、可复现实验评测闭环。真正“只是接框架”的部分反而比较外围，例如模型下载门面、Provider 切换、FAISS/BM25 的标准封装。
- [基于源码的推断] 当前最主要的工程风险不是“跑不起来”，而是“继续迭代时容易漂移”：双实现并存、`src/generation/answerer.py` 过重、缓存失效策略偏弱、benchmark 口径复杂、集成测试仍有空洞。

## 审查方法与证据口径

- [源码事实] 设计审查以本地源码为准，重点阅读了 `run_pipeline.py`、`run_prepare_benchmark.py`、`run_retrieval_eval.py`、`run_answer_eval.py`、`src/ingest/*`、`src/retrieval/*`、`src/generation/*`、`src/evaluation/*`、`scripts/ragctl.sh`、`scripts/remote_ops.py`。
- [远端运行事实 | 2026-04-13] 运行审查以远端 `/root/autodl-tmp/financial-rag-project` 为准，核验了：
  - `.venv/bin/python -m pytest -q`
  - `outputs/reports/retrieval_eval_summary.json`
  - `outputs/reports/answer_eval_summary_dev.json`
  - `outputs/reports/answer_eval_summary_full.json`
  - `outputs/reports/latency_summary_full.json`
  - `artifacts/remote_20260410_full_raw_repro_v2/reports/answer_eval_summary_full.json`
  - `artifacts/remote_20260410_full_raw_repro_v2/reports/latency_summary_full.json`
- [基于源码的推断] 下文所有结论都显式标注为三类之一：
  - `源码事实`
  - `远端运行事实`
  - `基于源码的推断`

## 第一章：项目在做什么

### 1.1 核心目标

- [源码事实] `run_pipeline.py` 负责从 PDF 生成 `data/chunks/chunks.jsonl`，`run_prepare_benchmark.py` 负责生成 retrieval/answer benchmark 素材，`run_retrieval_eval.py` 负责检索评测，`run_answer_eval.py` 负责答案生成与答案级评测。
- [源码事实] `src/ingest/pipeline.py:12` 的 `run_ingest_pipeline` 明确把中间产物落到 `parsed_pages`、`cleaned_pages`、`elements`、`chunks`、`reports`、`badcases`；`src/evaluation/output_archive.py:35` 和 `src/evaluation/run_metadata.py:91` 负责归档与元数据。
- [基于源码的推断] 因此这个项目的核心目标不是“聊天”，而是把金融研报 PDF 变成一个可检索、可引用、可拒答、可评测、可复现的证据驱动问答系统。
- [基于源码的推断] 它不是普通向量库 demo，原因有四个：
  - 先做 PDF 结构恢复，再做 chunk，不是直接抽纯文本。
  - 检索不是单路召回，而是 `Dense + BM25 + Hybrid + Rerank`。
  - 生成不是自由回答，而是绑定证据、输出 citation、做 support validation 和 abstain。
  - 整个系统自带 benchmark、badcase、artifact、metadata，而不是“能答一个样例就算完成”。

### 1.2 主要功能模块

| 模块 | 关键位置 | 主要职责 |
| --- | --- | --- |
| Ingest | `src/ingest/parser.py`、`src/ingest/cleaner.py`、`src/ingest/chunker.py`、`src/ingest/pipeline.py` | 解析 PDF、清洗噪声、构建 parent/child chunk、产出 ingest/chunk 报告 |
| Retrieval | `src/retrieval/embedder.py`、`src/retrieval/faiss_index.py`、`src/retrieval/bm25_index.py`、`src/retrieval/hybrid_retriever.py`、`src/retrieval/reranker.py`、`src/retrieval/runtime.py` | 构建/加载 dense 与 BM25 索引，执行 hybrid 检索和 cross-encoder rerank |
| Generation | `src/generation/prompt.py`、`src/generation/provider.py`、`src/generation/answerer.py`、`src/generation/citation_builder.py`、`src/generation/abstain.py` | 识别题型、挑选证据、组装 prompt、调用模型、修复 JSON、绑定 citation、决定是否拒答 |
| Evaluation | `src/evaluation/benchmark_assets.py`、`src/evaluation/retrieval_eval.py`、`src/evaluation/answer_eval.py`、`src/evaluation/run_metadata.py`、`src/evaluation/output_archive.py` | 准备 benchmark、物化 gold、计算 retrieval/answer 指标、沉淀 badcase、记录运行元数据 |
| Scripts / Ops | `scripts/ragctl.sh`、`scripts/remote_ops.py`、`bootstrap/*` | 封装标准入口、远端 SSH/SFTP 操作、环境检查 |

### 1.3 技术栈

| 层次 | 技术选型 | 证据 |
| --- | --- | --- |
| PDF 解析 | PyMuPDF | [源码事实] `src/ingest/parser.py` |
| 向量表示 | `BAAI/bge-m3` + `sentence-transformers` | [源码事实] `src/retrieval/embedder.py` |
| 稠密索引 | FAISS | [源码事实] `src/retrieval/faiss_index.py` |
| 稀疏索引 | `jieba` + `rank_bm25` | [源码事实] `src/retrieval/bm25_index.py` |
| 重排 | `BAAI/bge-reranker-v2-m3` | [源码事实] `src/retrieval/reranker.py` |
| 生成 | 本地 Hugging Face 模型 / DeepSeek API | [源码事实] `src/generation/provider.py:94` |
| 评测 | JSON/JSONL/Markdown 报告、badcase、metadata | [源码事实] `src/evaluation/*` |
| 运维 | Bash 封装、Paramiko 远端辅助 | [源码事实] `scripts/ragctl.sh`、`scripts/remote_ops.py` |

### 1.4 数据流、调用链、执行流程

#### 1.4.1 数据落盘链

```text
pdf/
  -> data/parsed_pages/parsed_pages.jsonl
  -> data/cleaned_pages/cleaned_pages.jsonl
  -> data/elements/elements.jsonl
  -> data/chunks/parent_chunks.jsonl
  -> data/chunks/child_chunks.jsonl
  -> data/chunks/chunks.jsonl
  -> outputs/indexes/*
  -> outputs/reports/*
  -> outputs/badcases/*
  -> artifacts/*
```

- [源码事实] `src/ingest/pipeline.py:25-31` 明确了 ingest 产物路径。
- [源码事实] `src/evaluation/output_archive.py:35` 把 `reports` 与 `badcases` 复制到 `artifacts`。

#### 1.4.2 主执行链

```text
run_pipeline.py
  -> src.ingest.pipeline.run_ingest_pipeline
  -> parse_pdf
  -> clean_pages
  -> build_parent_context_chunks
  -> build_chunk_strategies

run_prepare_benchmark.py
  -> prepare_benchmark_assets
  -> materialize_eval_set
  -> build_answer_seed_draft

run_retrieval_eval.py
  -> build_retrieval_runtime
  -> runtime.search
  -> evaluate_retrieval_methods

run_answer_eval.py
  -> resolve_benchmark_profile
  -> materialize_answer_eval_sets
  -> build_retrieval_runtime
  -> build_generation_answerer
  -> answerer.answer
  -> evaluate_answer_results
  -> archive_output_bundle / build_run_metadata
```

- [源码事实] `src/retrieval/runtime.py:197` 构建检索运行时，`src/retrieval/runtime.py:158` 执行单 query 的检索主链。
- [源码事实] `src/generation/answerer.py:2779` 是最终答案生成入口，`src/generation/answerer.py:1593` 负责准备证据，`src/generation/answerer.py:1846` 负责引用精修。
- [源码事实] `src/generation/abstain.py:211` 决定是否拒答；`src/evaluation/answer_eval.py:288` 做答案级评测。

## 第二章：项目审查

### 2.1 当前实现了什么

#### 2.1.1 Ingest

- [源码事实] `src/ingest/parser.py:132` 的 `_extract_sorted_blocks`、`src/ingest/parser.py:253` 的 `_build_page_elements`、`src/ingest/parser.py:315` 的 `parse_pdf` 已经实现了 PDF 块提取、页面元素构建、结构化输出。
- [源码事实] `src/ingest/chunker.py:17` 把生产策略固定为 `table_protected`；`src/ingest/chunker.py:570` 构建 parent context；`src/ingest/chunker.py:667` 同时生成 `fixed_window`、`title_aware`、`table_protected` 三套切块结果。
- [源码事实] `src/ingest/pipeline.py` 已将 ingest 过程组织成一个可反复执行的 pipeline，并产出 `ingest_summary.json` 与 `chunk_strategy_report.json`。

#### 2.1.2 Retrieval

- [源码事实] `src/retrieval/embedder.py:82` 支持 embedding 缓存构建与加载，`src/retrieval/model_store.py:9` 支持模型下载与缓存目录复用。
- [源码事实] `src/retrieval/hybrid_retriever.py:80` 定义了 `HybridRetriever`，使用 dense/bm25 融合、标题 overlap 加分、噪声惩罚。
- [源码事实] `src/retrieval/runtime.py:197` 将 dense、BM25、hybrid、rerank 组织为统一 runtime；`src/retrieval/runtime.py:158` 给出单次检索返回 `dense_rows`、`bm25_rows`、`hybrid_rows`、`rerank_rows` 以及 latency。

#### 2.1.3 Generation

- [源码事实] `src/generation/prompt.py:16` 已实现结构化 prompt 构造。
- [源码事实] `src/generation/provider.py:94` 支持本地模型与 DeepSeek 两种生成 provider。
- [源码事实] `src/generation/answerer.py:300`、`src/generation/answerer.py:331`、`src/generation/answerer.py:343` 已实现 answer mode 和 query domain 推断；`src/generation/answerer.py:1593` 负责证据准备；`src/generation/answerer.py:2779` 负责最终 answer。
- [源码事实] `src/generation/citation_builder.py:8` 负责 citation 绑定，`src/generation/abstain.py:211` 负责拒答决策。

#### 2.1.4 Evaluation

- [源码事实] `src/evaluation/benchmark_assets.py:231` 能从 chunks 生成 doc manifest；`src/evaluation/benchmark_assets.py:408` 能准备 retrieval seed；`src/evaluation/benchmark_assets.py:495` 能生成 answer seed draft。
- [源码事实] `src/evaluation/retrieval_eval.py:139` 已实现 retrieval 指标评估；`src/evaluation/answer_eval.py:288` 已实现 answer 指标评估。
- [源码事实] `src/evaluation/run_metadata.py:91` 记录 corpus、benchmark、source manifest 等元数据；`src/evaluation/output_archive.py:35` 和 `src/evaluation/output_archive.py:50` 负责归档与 scratch run policy。

#### 2.1.5 复现与运维

- [源码事实] `scripts/ragctl.sh:46` 和 `scripts/ragctl.sh:58` 分别封装了 `answer-eval-full` 与 `answer-eval-full-raw` 两种标准口径。
- [源码事实] `scripts/remote_ops.py` 使用 Paramiko 封装了 `install-key`、`upload`、`exec`、`download`。
- [远端运行事实 | 2026-04-13] 远端路径 `/root/autodl-tmp/financial-rag-project` 存在 `.venv`、`outputs`、`artifacts`、`tmp_stage6_data` 等完整目录，说明仓库不仅能本地跑，还已经过多轮远端复现。

### 2.2 远端已验证事实

#### 2.2.1 测试状态

- [远端运行事实 | 2026-04-13] 在远端 `/root/autodl-tmp/financial-rag-project` 上执行 `.venv/bin/python -m pytest -q`，结果为 `73 passed, 5 warnings in 6.77s`。
- [远端运行事实 | 2026-04-13] 远端 PATH 中没有 `python` / `python3` 的直接可用别名，但 `.venv/bin/python` 是有效入口。这解释了为什么标准执行入口应优先走 `ragctl.sh` 或显式 `.venv/bin/python`。

#### 2.2.2 检索评测

来源：`/root/autodl-tmp/financial-rag-project/outputs/reports/retrieval_eval_summary.json`

| 方法 | Query Count | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 48 | 0.5417 | 0.6458 | 0.3763 |
| BM25 | 48 | 0.5000 | 0.5833 | 0.3666 |
| Hybrid | 48 | 0.5417 | 0.7083 | 0.4000 |
| Hybrid+Rerank | 48 | 0.6667 | 0.6667 | 0.6038 |

- [远端运行事实 | 2026-04-13] 这个结果证明 rerank 后的排序质量提升明显，`MRR@10` 从 `0.4000` 提高到 `0.6038`。
- [基于源码的推断] 这组数据里 rerank 的主要价值在提升排序质量，而不是稳定扩大召回面；最直接的证据是 `MRR@10` 明显提升，而 `Recall@10` 没有同步提升。

#### 2.2.3 答案评测：`current-dev`

来源：`/root/autodl-tmp/financial-rag-project/outputs/reports/answer_eval_summary_dev.json`

| 口径 | Query Count | Answer Hit Rate | Citation Doc Hit Rate | Citation Span Hit Rate | Support Hit Rate | Fallback Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current-dev` | 16 | 0.7500 | 0.6250 | 0.5000 | 1.0000 | 0.1875 |

- [远端运行事实 | 2026-04-13] dev 集已达到“可持续迭代”的状态，但 citation span 仍明显弱于 doc-level citation。

#### 2.2.4 答案评测：`historical-full-core`

来源：`/root/autodl-tmp/financial-rag-project/outputs/reports/answer_eval_summary_full.json`

| 口径 | Query Count | Answer Hit Rate | Citation Doc Hit Rate | Citation Span Hit Rate | Support Hit Rate | Abstain Precision | Abstain Recall | Fallback Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `historical-full-core` | 40 | 0.6857 | 0.8286 | 0.4000 | 0.9143 | 0.8333 | 1.0000 | 0.1500 |

- [远端运行事实 | 2026-04-13] full-core 的 doc-level citation 很强，但 span-level citation 仍不稳。
- [远端运行事实 | 2026-04-13] `Abstain Recall = 1.0000` 说明该版本已经能识别“必须拒答”的题，不再是无脑硬答。

#### 2.2.5 答案评测：`historical-full-raw`

来源：`/root/autodl-tmp/financial-rag-project/artifacts/remote_20260410_full_raw_repro_v2/reports/answer_eval_summary_full.json`

| 口径 | Query Count | Answer Hit Rate | Citation Doc Hit Rate | Citation Span Hit Rate | Support Hit Rate | Abstain Precision | Abstain Recall | Fallback Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `historical-full-raw` | 50 | 0.5778 | 0.6444 | 0.3111 | 0.9333 | 0.8333 | 1.0000 | 0.1600 |

- [远端运行事实 | 2026-04-13] `historical-full-raw` 的难度显著高于 `historical-full-core`，最明显的退化出现在 `Answer Hit Rate` 与 `Citation Span Hit Rate`。
- [基于源码的推断] 这说明系统的主瓶颈不是“能不能召回某个相关文档”，而是复杂 comparison / inductive 问题的多文档表达与精确证据对齐。

#### 2.2.6 延迟

来源：
- `/root/autodl-tmp/financial-rag-project/outputs/reports/latency_summary_full.json`
- `/root/autodl-tmp/financial-rag-project/artifacts/remote_20260410_full_raw_repro_v2/reports/latency_summary_full.json`

| 口径 | Query Count | Avg Retrieval Latency (ms) | Avg Rerank Latency (ms) | Avg Generation Latency (ms) | End-to-End Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `historical-full-core` | 40 | 62.35 | 355.58 | 5781.89 | 6239.73 |
| `historical-full-raw` | 50 | 54.43 | 306.23 | 7732.25 | 8127.83 |

- [远端运行事实 | 2026-04-13] 生成阶段明显是主耗时项，远高于检索与 rerank。
- [基于源码的推断] 如果后续产品化，优先优化 generation context 长度、比较/归纳题的多阶段调用成本，会比继续优化 FAISS 更有收益。

### 2.3 设计上的优点

1. [源码事实] **结构化 PDF ingest 不是表面工作。** `src/ingest/parser.py:132`、`src/ingest/parser.py:253`、`src/ingest/parser.py:315` 已经把 PDF 从页面块恢复为有结构的元素，而不是直接把全文本塞给 chunker。
2. [源码事实] **chunk 设计有明显领域意识。** `src/ingest/chunker.py:17` 固定生产策略为 `table_protected`，说明系统明确把表格、图注、正文的边界当作检索质量问题处理。
3. [源码事实] **检索链是分阶段设计的。** `src/retrieval/runtime.py:197` + `src/retrieval/hybrid_retriever.py:80` + `src/retrieval/runtime.py:158` 体现了 recall 与 precision 分离的工程思路。
4. [源码事实] **生成链有证据约束。** `src/generation/answerer.py:1593`、`src/generation/citation_builder.py:8`、`src/generation/abstain.py:211` 说明系统不是“top-k 拼 prompt”，而是显式做证据选择、citation、拒答。
5. [源码事实] **评测闭环完整。** `src/evaluation/benchmark_assets.py:408`、`src/evaluation/retrieval_eval.py:139`、`src/evaluation/answer_eval.py:288`、`src/evaluation/run_metadata.py:91` 组成了“可回归、可定位、可归档”的闭环。
6. [远端运行事实 | 2026-04-13] **这套设计不是纸面设计。** 远端已有多轮 `dev/full-core/full-raw` 归档，且当前测试全通过。

### 2.4 问题、风险、坏味道、性能瓶颈、可维护性问题

#### 高优先级

1. **双实现并存，存在长期漂移风险。**  
   - [源码事实] 入口和测试主要使用 `src/retrieval/runtime.py`、`src/generation/answerer.py`，但仓库中同时还保留了旧版 `src/runtime.py`、`src/answerer.py`。  
   - [基于源码的推断] 后续修 bug 时极易出现“改了一套，另一套继续漂”的问题。  
   - 文件定位：`src/runtime.py`、`src/answerer.py`、`src/retrieval/runtime.py`、`src/generation/answerer.py`。

2. **`src/generation/answerer.py` 职责过载。**  
   - [源码事实] 同一文件同时处理题型识别、域路由、证据选择、prompt 构造、JSON 修复、citation refinement、fallback、abstain。核心节点集中在 `src/generation/answerer.py:1593`、`1846`、`2779`。  
   - [基于源码的推断] 这会让任何规则改动都影响整条回答链，回归成本随规则数量增长而加速上升。  
   - 文件定位：`src/generation/answerer.py`。

#### 中优先级

3. **缓存失效策略偏弱。**  
   - [源码事实] `src/retrieval/model_store.py:9` 只要模型目录存在且非空就复用；`src/retrieval/embedder.py:82` 只检查 `model_name`、`chunk_ids_hash`、`chunk_text_hash`、`embedding_dim`。  
   - [基于源码的推断] 如果同名模型权重被更新、tokenizer 变化、目录损坏但未完全空掉，系统可能静默复用旧缓存。  
   - 文件定位：`src/retrieval/model_store.py`、`src/retrieval/embedder.py`、`src/retrieval/faiss_index.py`、`src/retrieval/bm25_index.py`。

4. **本地生成参数“可配但不生效”。**  
   - [源码事实] `src/generation/provider.py:94` 将 `temperature` / `top_p` 传入本地 answerer；但本地生成实际在 `src/generation/answerer.py` 的 `_generate` 中固定贪心生成。  
   - [基于源码的推断] 这会给调参者造成误导，以为本地生成分支支持采样控制，实际上可能并没有。  
   - 文件定位：`src/generation/provider.py`、`src/generation/answerer.py`。

5. **benchmark 口径复杂，容易被外部误读。**  
   - [源码事实] `run_answer_eval.py:53` 会按 split 推导 benchmark profile；`scripts/ragctl.sh:46` 和 `scripts/ragctl.sh:58` 又把 `full-core` 和 `full-raw` 作为不同命令封装。  
   - [远端运行事实 | 2026-04-13] 当前远端 `outputs/reports/answer_eval_summary_full.json` 是 `historical-full-core` 40 题，而不是 `historical-full-raw` 50 题。  
   - [基于源码的推断] 如果对外不明确标注 profile，极容易把不同版本的“full”指标混为一谈。  
   - 文件定位：`run_answer_eval.py`、`scripts/ragctl.sh`。

6. **集成测试仍有空洞。**  
   - [源码事实] 当前 `tests/*` 更偏纯函数和规则层，覆盖了 routing、quality rules、abstain、cache、metadata。  
   - [基于源码的推断] 但真实 PDF ingest、真实模型生成、远端 provider、整链 CLI 入口仍缺少强约束的集成测试。  
   - 文件定位：`tests/test_answerer_routing.py`、`tests/test_answerer_quality_rules.py`、`tests/test_runtime_abstain.py`、`tests/test_retrieval_cache.py`。

#### 低优先级但需要持续关注

7. **性能瓶颈明显集中在 generation。**  
   - [远端运行事实 | 2026-04-13] full-core 平均生成延迟 `5781.89ms`，full-raw 平均生成延迟 `7732.25ms`，远高于检索与 rerank。  
   - [基于源码的推断] 后续如果要产品化，需要优先优化复杂题型的上下文裁剪和生成轮次。

8. **citation span 明显弱于 doc-level citation。**  
   - [远端运行事实 | 2026-04-13] full-core 的 `Citation Doc Hit Rate = 0.8286`，但 `Citation Span Hit Rate = 0.4000`；full-raw 分别是 `0.6444` 与 `0.3111`。  
   - [基于源码的推断] 当前系统已经能“找到对的文档”，但“把答案锚定到足够精确的证据片段”仍是短板。

### 2.5 哪些部分体现了真正的技术含量

1. [源码事实] **结构感知 PDF 解析与 chunk 设计。**  
   关键位置：`src/ingest/parser.py`、`src/ingest/chunker.py`。  
   [基于源码的推断] 这里体现的是对金融研报文档结构的理解，而不是简单库调用。

2. [源码事实] **Hybrid+Rerank 检索链。**  
   关键位置：`src/retrieval/hybrid_retriever.py:80`、`src/retrieval/runtime.py:158`。  
   [基于源码的推断] 融合、标题重叠、噪声惩罚、rerank 组合在一起，说明作者在围绕“金融研报问法”做有意识的排序工程。

3. [源码事实] **证据选择、citation、abstain。**  
   关键位置：`src/generation/answerer.py:1593`、`1846`、`2779`，`src/generation/citation_builder.py:8`，`src/generation/abstain.py:211`。  
   [基于源码的推断] 这是整个项目区别于“普通 RAG demo”的核心。

4. [源码事实] **benchmark、badcase、metadata、artifact 闭环。**  
   关键位置：`src/evaluation/*`。  
   [基于源码的推断] 这部分的工程成熟度高，决定了系统能不能持续演进，而不只是“一次性做出来”。

### 2.6 哪些部分主要是调用现成框架、缺少核心思考

1. [源码事实] `src/retrieval/model_store.py` 本质上是 `modelscope.snapshot_download` 的包装层。  
2. [源码事实] `src/retrieval/embedder.py` 的核心语义编码能力来自 `SentenceTransformer`，项目侧主要做了缓存和元数据封装。  
3. [源码事实] `src/retrieval/faiss_index.py` 与 `src/retrieval/bm25_index.py` 更偏标准索引封装，而不是自研检索算法。  
4. [源码事实] `src/generation/provider.py` 主要在做“本地模型 / DeepSeek API”的 provider 选择，不是生成策略本身。  
5. [基于源码的推断] 这些模块并不低价值，但它们的价值更偏工程集成，不应被包装成“核心算法创新”。

## 第三章：从项目反推的知识点树

### 3.1 基础知识点

- `PDF 版面重建`：`src/ingest/parser.py:132`、`253`、`315`
- `文本归一化与标题清洗`：`src/utils/text_utils.py`、`src/evaluation/benchmark_assets.py`
- `行业 / 来源识别`：`src/utils/text_utils.py`、`src/evaluation/benchmark_assets.py`
- `BM25 稀疏检索`：`src/retrieval/bm25_index.py`
- `FAISS 稠密索引`：`src/retrieval/faiss_index.py`

### 3.2 核心知识点

- `分层 chunk 与 table_protected 策略`：`src/ingest/chunker.py:17`、`570`、`667`
- `Dense Embedding 缓存与复用`：`src/retrieval/embedder.py:82`
- `Hybrid 检索与 Rerank`：`src/retrieval/hybrid_retriever.py:80`、`src/retrieval/runtime.py:158`
- `结构化 Prompt`：`src/generation/prompt.py:16`
- `答案生成主链`：`src/generation/answerer.py:2779`

### 3.3 进阶知识点

- `Query 意图识别与 domain routing`：`src/generation/answerer.py:300`、`331`、`343`
- `多文档证据选择`：`src/generation/answerer.py:1593`
- `Citation 绑定与证据精修`：`src/generation/answerer.py:1846`、`src/generation/citation_builder.py:8`
- `Abstain / Support Validation`：`src/generation/abstain.py:211`
- `复杂 benchmark 口径管理`：`run_answer_eval.py:53`、`scripts/ragctl.sh:46`、`58`

### 3.4 工程化知识点

- `benchmark materialization`：`src/evaluation/benchmark_assets.py:408`、`495`
- `retrieval/answer 分层评测`：`src/evaluation/retrieval_eval.py:139`、`src/evaluation/answer_eval.py:288`
- `run metadata 与 source manifest`：`src/evaluation/run_metadata.py:91`
- `artifact 归档与 scratch run policy`：`src/evaluation/output_archive.py:35`、`50`
- `缓存失效与回归测试`：`src/retrieval/embedder.py:82`、`tests/test_retrieval_cache.py`

### 3.5 面试高频知识点

- 为什么要 `Hybrid Retrieval + Rerank`
- 为什么要做 `parent/child chunk`，而不是固定窗口
- citation 是怎么做的，为什么 citation 比 answer 更难
- abstain 怎么设计，为什么不能只看模型置信度
- RAG 应该如何评估，为什么不能只看最终回答对错
- 如何证明一个 RAG 系统是“可复现、可迭代”的

## 第四章：每个知识点的完整讲法

### 4.1 PDF 版面重建

- 分类：基础知识点
- 具体位置：`src/ingest/parser.py:132` 的 `_extract_sorted_blocks`，`src/ingest/parser.py:253` 的 `_build_page_elements`，`src/ingest/parser.py:315` 的 `parse_pdf`
- 解决的问题：原始 PDF 抽文本常常乱序、双栏串读、标题层级丢失、图表和正文混杂，直接切文本会把后续检索和引用全部带偏。
- 为什么这样设计：因为项目的上游数据不是结构化 HTML，而是金融研报 PDF；如果不先恢复阅读顺序和页面结构，后面的 chunk、citation、span 对齐都会失真。
- 底层原理：基于页面 block 的 `bbox`、列位置、字号、粗体、标题栈等启发式规则，把页面恢复成更接近人类阅读顺序的元素序列。
- 可替代方案：纯 `pdfplumber` 文本抽取、OCR、版面理解模型（如 LayoutLM / DocTR）或商业解析服务。
- trade-off：规则法快、可控、成本低、便于调试，但跨模板泛化不如模型法；模型法泛化更好，但依赖更重、可解释性更差。
- 面试表达模板：`这个项目里我没有把 PDF 当成普通文本源，而是先做版面重建，把页面恢复成有阅读顺序的结构化元素。这样后面的 chunk、引用和 span 评测才有意义，否则检索质量会被上游噪声直接拖垮。`

### 4.2 文本归一化与标题/行业识别

- 分类：基础知识点
- 具体位置：`src/utils/text_utils.py`，`src/evaluation/benchmark_assets.py` 中的 `short_title_from_file_name`、`industry_from_file_name`
- 解决的问题：研报文件名、标题、行业标签和噪声字符串并不稳定；如果不归一化，标题匹配、行业路由、benchmark 物化都容易出错。
- 为什么这样设计：这个项目大量依赖“报告名”“行业桶”“标题片段”作为弱监督和路由特征，所以先做 canonicalization 是必要的。
- 底层原理：字符串标准化、正则清洗、关键词映射、文件名模式解析。
- 可替代方案：训练分类模型、维护人工元数据表、外部知识库映射。
- trade-off：规则法维护成本低、解释性好，但依赖文件命名约定；分类模型泛化更强，但需要标注和推理成本。
- 面试表达模板：`我把文件名和正文里的不稳定表示统一成 canonical form，再拿它做标题重叠、行业路由和 benchmark 物化。这个步骤看起来简单，但对后续规则系统非常关键。`

### 4.3 分层 chunk 与表格保护

- 分类：核心知识点
- 具体位置：`src/ingest/chunker.py:17`、`src/ingest/chunker.py:570`、`src/ingest/chunker.py:667`
- 解决的问题：固定窗口切块会把表格、图注、正文边界打碎，导致数字问答和 citation 极不稳定。
- 为什么这样设计：金融研报里最敏感的信息经常在表格、图表说明、段落上下文中，如果只追求“切得均匀”，检索粒度和解释粒度会冲突。
- 底层原理：把检索粒度和上下文粒度拆开，生成 parent chunk 和 child chunk；同时为表格保留专门的 `table_protected` 策略。
- 可替代方案：固定窗口、纯 token window、按标题硬切、不做 parent/child。
- trade-off：分层 chunk 会增加索引复杂度和实现复杂度，但能显著提升复杂证据的可用性。
- 面试表达模板：`我没有把所有 chunk 平权处理，而是把 parent context 和 child evidence 拆开。这样检索时可以精确命中，生成时又能补足上下文，尤其适合表格和跨段证据。`

### 4.4 Dense Embedding + FAISS 缓存索引

- 分类：核心知识点
- 具体位置：`src/retrieval/embedder.py:82`、`src/retrieval/faiss_index.py`、`src/retrieval/model_store.py:9`
- 解决的问题：需要做语义召回，同时避免每次评测都重复构建 embedding 和向量索引。
- 为什么这样设计：RAG 迭代频繁，如果每次都重新编码几千个 chunk，会把实验反馈周期拉长。
- 底层原理：使用归一化向量做内积检索，近似余弦相似度；用 `chunk_ids_hash`、`chunk_text_hash`、`model_name` 等元数据判断缓存是否可复用。
- 可替代方案：Milvus、HNSW、Annoy、Elasticsearch vector、全量每次重建。
- trade-off：当前方案简单准确、易于本地和单机部署，但规模继续上去时内存与索引时延会成为约束。
- 面试表达模板：`我在项目里不只做向量检索，还专门处理了缓存失效问题。重点不是“有没有 embeddings.npy”，而是“这份 embedding 是否对应当前模型和当前 chunk 内容”。`

### 4.5 BM25 稀疏检索

- 分类：基础知识点
- 具体位置：`src/retrieval/bm25_index.py`
- 解决的问题：数字、专有名词、标题词、行业术语这类精确词，单靠 dense embedding 很容易召回不稳。
- 为什么这样设计：金融问答有大量“同比多少”“哪份报告提到 X”“某个术语在哪出现”这类 lexical-sensitive 查询。
- 底层原理：分词、词频、逆文档频率、BM25 打分。
- 可替代方案：Lucene/Elasticsearch、SPLADE、完全依赖 dense。
- trade-off：BM25 对精确词很强，但对语义泛化差；dense 则反过来。两者组合比单用其中一个稳得多。
- 面试表达模板：`BM25 在这个项目里不是备胎，而是 dense 的互补项。对数字、标题词、报告名这类关键词，BM25 往往比纯语义召回更可靠。`

### 4.6 Hybrid Retrieval + Rerank

- 分类：核心知识点 / 面试高频
- 具体位置：`src/retrieval/hybrid_retriever.py:80`、`src/retrieval/runtime.py:158`
- 解决的问题：dense 和 BM25 各有盲点，直接用单路召回或单次排序容易错失最佳证据。
- 为什么这样设计：检索链通常需要先解决“召回面”，再解决“排序精度”；一个模型同时把两件事都做到最好，往往成本更高。
- 底层原理：先做 dense 与 BM25 的 min-max 归一化融合，再加入标题重叠和噪声惩罚，最后使用 cross-encoder rerank 提升 top-k 精度。
- 可替代方案：只用 dense、只用 BM25、只用 rerank、端到端 LLM 检索。
- trade-off：效果更稳，但 rerank 会带来明显时延；远端指标也证明 rerank 的主要价值在排序而不是召回。
- 面试表达模板：`我把检索拆成 recall 和 precision 两阶段。先用 dense+BM25 扩大候选，再用 reranker 把真正有用的证据排到前面，这比单路检索稳定得多。`

### 4.7 Query 意图路由与 domain priority

- 分类：进阶知识点
- 具体位置：`src/generation/answerer.py:300`、`src/generation/answerer.py:331`、`src/generation/answerer.py:343`，`run_answer_eval.py` 中的 `apply_retrieval_domain_priority`
- 解决的问题：事实问句、对比问句、归纳问句、报告定位问句对证据数量、来源多样性、答案格式的要求完全不同。
- 为什么这样设计：如果所有 query 走一条检索和生成路径，复杂题型会过拟合简单题型的策略，最终导致 citation 和 support validation 双双失真。
- 底层原理：基于关键词、模式匹配和领域词表做轻量 query classification，再按 domain bucket 调整候选优先级。
- 可替代方案：训练专门的 query router、让 LLM 自己判断、完全不做路由。
- trade-off：规则路由可解释、便宜、易控，但覆盖面依赖人工维护；学习型 router 泛化更强，但调试成本更高。
- 面试表达模板：`我没有让所有 query 走同一条 path，而是先识别它是 fact、comparison、inductive 还是 report lookup，再决定证据选择和回答约束。`

### 4.8 多文档证据选择与 domain guard

- 分类：进阶知识点
- 具体位置：`src/generation/answerer.py:1593`
- 解决的问题：comparison 和 inductive 题需要多文档、多来源证据；如果只拿单文档 top-k，回答很容易看起来合理但其实缺少对比基础。
- 为什么这样设计：多文档问答最难的不是“找到一段相关文本”，而是“找到一组相互可比较、且不串域的证据”。
- 底层原理：构建有顺序的候选池，对文档去重、来源多样性、领域一致性进行约束，然后再决定最终 `used_evidence_ids`。
- 可替代方案：直接取 rerank top-k、让模型自己从 evidence blocks 中挑、用图结构聚合。
- trade-off：当前方案更保守，但显著降低了串域和伪比较问题；代价是实现复杂、规则较多。
- 面试表达模板：`在多文档题里，我把证据选择单独做成了一层约束逻辑，而不是把 top-k 全塞给模型。这样能显著减少跨行业、跨语境误答。`

### 4.9 结构化 Prompt + JSON 修复

- 分类：进阶知识点
- 具体位置：`src/generation/prompt.py:16`、`src/generation/answerer.py` 的 JSON 解析与修复链
- 解决的问题：LLM 输出容易格式漂移、字段缺失、把答案和引用混在一起，导致后处理不稳定。
- 为什么这样设计：项目需要对 `final_answer`、`evidence_summary`、`uncertainty_note`、`used_evidence_ids` 分别做评测与校验，所以必须先得到结构化中间表示。
- 底层原理：通过受控 schema 限制生成格式，再通过解析和修复函数最大化恢复结构化字段。
- 可替代方案：完全自由文本输出后再正则抽取、工具调用、函数调用协议。
- trade-off：schema 化更稳，但 prompt 更复杂；自由文本更自然，但几乎不利于严谨评测。
- 面试表达模板：`我没有直接让模型输出最终答案，而是让它先输出结构化 JSON 骨架。这样 citation、support validation、fallback 都能基于同一份中间表示执行。`

### 4.10 Citation 绑定与证据精修

- 分类：进阶知识点 / 面试高频
- 具体位置：`src/generation/citation_builder.py:8`、`src/generation/answerer.py:1846`
- 解决的问题：答案可能答对，但引用错页、错段、错文档；这在金融场景里比答错更危险。
- 为什么这样设计：项目目标是“证据驱动问答”，citation 不是装饰，而是答案可信度的外显接口。
- 底层原理：先按 `used_evidence_ids` 建 citation，再优先选同 section、同页面、同 body-support 的 chunk 进行精修。
- 可替代方案：直接把 top-k evidence 原样附在答案后面、只做 doc-level citation、不做 citation。
- trade-off：citation 精修增加了实现复杂度，但让系统从“能回答”升级到“能审计”。
- 面试表达模板：`我会把 citation 当成一等公民来设计。答对只是第一步，更重要的是能把答案锚定到具体证据片段，否则系统很难在严肃场景落地。`

### 4.11 Abstain 与 Support Validation

- 分类：进阶知识点 / 面试高频
- 具体位置：`src/generation/abstain.py:211`
- 解决的问题：模型可能在证据不足、数值不匹配、来源不够、证据冲突时仍然给出貌似合理的答案。
- 为什么这样设计：金融问答场景里，错误地自信回答往往比拒答更危险，尤其是数值题和归纳题。
- 底层原理：综合检查 `missing_numeric_tokens`、`conflict_detected`、`source_count`、题型要求、fallback reason 等信号，再输出 abstain 决策。
- 可替代方案：只看检索分数、只看模型自评、只在 prompt 里要求“如果不确定就说不知道”。
- trade-off：拒答机制会牺牲一部分覆盖率，但显著提升可靠性与可控性。
- 面试表达模板：`我在这个项目里把拒答设计成证据层规则，而不是情绪化的“模型说不确定”。只要证据不支撑，我宁可拒答，也不输出 unsupported answer。`

### 4.12 Retrieval / Answer Eval 闭环

- 分类：工程化知识点 / 面试高频
- 具体位置：`src/evaluation/benchmark_assets.py:408`、`src/evaluation/retrieval_eval.py:139`、`src/evaluation/answer_eval.py:288`
- 解决的问题：如果只看几个样例，系统很难判断到底是检索错了、生成错了、citation 错了，还是 benchmark 本身有偏差。
- 为什么这样设计：RAG 需要分层定位问题，不能把所有问题都归咎于大模型。
- 底层原理：先把 seed 物化为 gold chunk/doc，再分别计算 retrieval 指标和 answer/citation/support/abstain 指标。
- 可替代方案：人工 spot check、只看最终准确率、A/B chat 演示。
- trade-off：体系化评测维护成本高，但它是系统长期迭代的基础设施。
- 面试表达模板：`我评估 RAG 不只看最后答没答对，而是把 retrieval、citation、support、abstain 分开评。这样定位优化方向时不会只会盲调 prompt。`

### 4.13 Run Metadata 与 Artifact 可复现

- 分类：工程化知识点
- 具体位置：`src/evaluation/run_metadata.py:91`、`src/evaluation/output_archive.py:35`、`50`
- 解决的问题：实验结果容易和代码版本、语料版本、benchmark 版本脱钩，导致“指标看起来提升了，但不知道提升来自哪里”。
- 为什么这样设计：这个项目明显在做持续迭代，如果没有 source manifest 和 artifact 归档，过几轮之后历史结果就不可比了。
- 底层原理：对 corpus、benchmark、source manifest 做内容签名，并把结果、badcase、metadata 一起归档。
- 可替代方案：只保留 Markdown 报告、只看日志时间戳、手工记录版本。
- trade-off：元数据和归档会增加目录复杂度，但换来可追溯性和团队协作稳定性。
- 面试表达模板：`我不会只留一份 summary.md，而是把这次运行对应的 corpus、benchmark、source manifest 和结果一起归档。这样任何一次指标变化都能回溯到具体输入。`

### 4.14 缓存失效与回归测试

- 分类：工程化知识点
- 具体位置：`src/retrieval/embedder.py:82`、`src/retrieval/model_store.py:9`、`tests/test_retrieval_cache.py`、`tests/test_answerer_routing.py`、`tests/test_answerer_quality_rules.py`、`tests/test_runtime_abstain.py`
- 解决的问题：规则系统和缓存系统最怕“悄悄错”；一旦缓存过期失效判断不准确，或者规则改动没有测试护栏，结果会漂得很隐蔽。
- 为什么这样设计：这个项目的核心逻辑大量依赖规则和组合策略，必须靠测试把关键行为钉住。
- 底层原理：通过内容哈希判断缓存是否可复用，通过回归测试约束 routing、abstain、citation、profile 行为。
- 可替代方案：每次全量重建、完全手工回归、只做少量 smoke test。
- trade-off：测试和缓存元数据会增加维护量，但比“系统静默漂移”安全得多。
- 面试表达模板：`这类启发式系统最怕静默退化，所以我会同时做两件事：一是让缓存失效基于内容而不是时间戳，二是把核心规则写成回归测试。`

## 最终判断

- [基于源码的推断] 这是一个已经跨过“能不能做出来”阶段、进入“如何把准确性、可解释性和可维护性继续做稳”的项目。
- [基于源码的推断] 如果从学习价值看，最值得深入吸收的是：`PDF 结构化 ingest`、`Hybrid+Rerank`、`证据约束生成`、`citation/abstain`、`评测闭环`。
- [基于源码的推断] 如果从工程风险看，最值得尽快治理的是：`双实现并存`、`answerer 过重`、`缓存失效策略`、`benchmark 口径复杂`、`集成测试空洞`。
- [基于源码的推断] 如果从面试看，这个项目最强的叙事不是“我用了 FAISS / BGE / DeepSeek”，而是“我把金融研报 PDF 做成了可检索、可引用、可拒答、可评测、可复现的证据驱动 RAG 系统”。这才是真正能拉开差距的地方。
