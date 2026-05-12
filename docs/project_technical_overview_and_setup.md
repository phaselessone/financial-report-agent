# 金融研报 RAG 项目技术说明与搭建说明

## 1. 项目定位

这个项目不是一个通用聊天机器人，而是一套面向金融/行业研报 PDF 的端到端 RAG 系统。目标是把大量结构复杂、更新频繁、带表格和标题层级的研究报告，转成可检索、可引用、可评测的问答系统。

当前系统已经覆盖四条主线：

1. PDF 解析、清洗、分块与中间产物落盘
2. Dense / BM25 / Hybrid / Rerank 检索链路
3. 基于证据的答案生成、引用绑定、支持性校验与拒答
4. 检索级与答案级评测、坏例收集、运行归档

代码主入口位于：

- `run_pipeline.py`：PDF ingest 与 chunk 产物生成
- `run_prepare_benchmark.py`：评测物料准备
- `run_retrieval_eval.py`：检索评测
- `run_answer_eval.py`：答案生成与答案级评测

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

完整数据流为：

`pdf/` -> `data/parsed_pages` -> `data/cleaned_pages` -> `data/elements` -> `data/chunks` -> `outputs/indexes` / `outputs/reports` / `outputs/badcases` -> `artifacts/`

历史压力测试则使用 stage6 语料：

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

## 5. 技术选型与理由

## 5.1 当前实际采用的技术

| 层级 | 方案 |
|---|---|
| PDF 解析 | PyMuPDF |
| Dense Embedding | `BAAI/bge-m3` |
| Dense Index | FAISS |
| Sparse Retrieval | BM25 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM Provider | DeepSeek API 或本地 `Qwen/Qwen2.5-7B-Instruct` |
| 评测输出 | JSON / JSONL / Markdown |
| 远端运维 | `scripts/remote_ops.py` + Paramiko |

## 5.2 与图片中的常见方案关系

你给的图片里提到的 RAGFlow、FAISS、FlagEmbedding、LangChain-ChatChat、QAnything、LlamaIndex、Milvus、GraphRAG，更适合放在“参考路线”里介绍，而不是直接说当前项目用了全部这些组件。

更准确的说法是：

1. 当前项目借鉴了成熟 RAG 工程里的思路，但实现上偏“轻量自建链路”
2. 真正落地的核心依赖主要是 PyMuPDF、FAISS、BGE embedding/reranker、DeepSeek/Qwen
3. 暂未引入 Milvus、Elasticsearch、LangChain 编排或 GraphRAG 复杂图检索，这样做是为了先把 PDF ingest、检索质量、证据约束和评测闭环打稳

如果在面试里被问“为什么不用 Milvus / LangChain / GraphRAG”，可以直接答：

1. 当前语料规模还没逼到必须上分布式向量库
2. 项目优先解决 PDF 结构化、混合检索和可追溯评测
3. 先把单机版本闭环做实，比堆中间件更有价值

## 6. 当前已验证的工程状态

以下结论基于本地代码、远端部署实例 `/root/autodl-tmp/financial-rag-project`、远端 summary 文件，以及远端测试结果。

### 6.1 环境与测试

1. 远端健康检查通过：Python、依赖、DeepSeek 配置、评测种子均可用
2. 远端测试使用项目标准入口 `python -m pytest -q`，结果为 `73 passed`
3. 当前远端部署路径为 `/root/autodl-tmp/financial-rag-project`
4. 远端存在独立 `.venv`，已配置 `.env.deepseek` / `.env.runtime`

### 6.2 数据与 chunk

历史文档记录的基线规模为：

- PDF：54 份
- 页数：680
- 生产 chunk：4239
- ingest badcase：10

这些数字来自项目复盘文档，说明 ingest 端已经不是空壳，而是有实打实的语料和坏例管理。

### 6.3 检索结果

远端当前 `outputs/reports/retrieval_eval_summary.json` 显示，在 48 条检索评测题上：

| 方法 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| Dense | 0.5417 | 0.6458 | 0.3763 |
| BM25 | 0.5000 | 0.5833 | 0.3666 |
| Hybrid | 0.5417 | 0.7083 | 0.4000 |
| Hybrid+Rerank | 0.6667 | 0.6667 | 0.6038 |

结论很明确：`Hybrid+Rerank` 仍然是当前最强的生产检索入口。

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

这说明当前版本在当前语料的开发集上已经达到“可以继续迭代”的状态。

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

这个结果说明：

1. 在历史核心集上，系统已经不是“跑不通”，而是进入了精修阶段
2. 当前最大短板不是 doc-level citation，而是 span-level citation 和复杂 comparison / inductive 表达
3. 拒答策略已经开始变得有用，不再是完全失控

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

### 7.3 裸跑 `pytest` 存在入口陷阱

项目标准回归命令是 `python -m pytest -q`，而不是裸 `pytest -q`。前者在远端通过，后者会因为导入路径问题在收集阶段失败。这类问题不影响主链路，但会影响新同学搭环境和 CI 接入体验。

### 7.4 生成延迟仍是主要耗时项

检索延迟通常在毫秒级，但完整答案生成仍明显慢于检索，后续如果要进一步产品化，需要继续控制：

1. 上下文长度
2. comparison / inductive 多阶段调用成本
3. fallback 与 abstain 带来的额外校验成本

## 8. 本地搭建说明

## 8.1 环境要求

推荐环境：

1. Python 3.12
2. Linux / WSL / Git Bash
3. CUDA GPU（检索 embedding、reranker、本地生成时更合适）
4. 如果只走 DeepSeek API，可不依赖本地生成模型

核心依赖见 `requirements.txt`：

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
- pytest

## 8.2 创建虚拟环境

Linux / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果需要远端运维脚本，再额外安装：

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

如果要跑历史 full core / raw 压力集，需要确认 stage6 历史语料已准备好，例如：

```text
tmp_stage6_data/chunks/chunks.jsonl
data/eval_set/answer_eval_seed_full.jsonl
```

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
  --chunks-path tmp_stage6_data/chunks/chunks.jsonl \
  --data-dir tmp_stage6_data \
  --corpus-label stage6-historical \
  --answer-seed-full-path data/eval_set/answer_eval_seed_full.jsonl
```

或：

```bash
bash scripts/ragctl.sh answer-eval-full
```

### F. 跑测试

推荐使用：

```bash
python -m pytest -q
```

如果只是按 `Makefile` 走：

```bash
make test
```

## 9. 远端搭建与审查结果

本次审查确认，远端当前部署信息如下：

- 主机：`connect.westb.seetacloud.com:25485`
- 用户：`root`
- 部署目录：`/root/autodl-tmp/financial-rag-project`
- 虚拟环境：`/root/autodl-tmp/financial-rag-project/.venv`

远端已验证：

1. `bash bootstrap/check_env.sh` 通过
2. `python -m pytest -q` 通过，`73 passed`
3. 远端保留了从 `tmp_stage3_*` 到 `tmp_stage15_*` 的多轮实验目录
4. 远端 `artifacts/` 中保留了 4 月 10 日的多轮 dev / full repro 产物

说明这套项目不只是“本地代码堆着”，而是真正有部署、有回归、有阶段性实验记录。

## 10. 面试/答辩时怎么介绍这个项目

可以按下面这条主线讲：

1. 业务问题：金融研报 PDF 结构复杂、更新快、必须可引用，所以单纯微调不合适，先做 RAG
2. 数据工程：我不是直接把 PDF 丢给模型，而是自己做了解析、清洗、标题感知和表格保护 chunk
3. 检索工程：不是单纯向量检索，而是 Dense + BM25 的混合召回，再用 reranker 做最终排序
4. 生成工程：对 comparison / inductive 做了分题型处理，并且加了证据绑定、支持性校验和拒答
5. 评测工程：不仅有 Recall/MRR，还有 answer hit、citation hit、support hit、abstain 和 badcase 归档
6. 工程结论：dev 集已经达到可持续迭代状态，历史 core 压力集具备可用性，但复杂题型和 span-level citation 仍需继续优化

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
