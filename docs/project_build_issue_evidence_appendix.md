# 证据附录

## A. 指标汇总表

### A.1 Ingest / Chunk

| 指标 | 数值 | 证据来源 |
|---|---:|---|
| PDF 数量 | 54 | `data/reports/ingest_summary.json` |
| 页数 | 680 | `data/reports/ingest_summary.json` |
| 生产 Chunk 数量 | 4239 | `data/reports/ingest_summary.json` |
| Ingest badcase | 10 | `data/reports/ingest_summary.json` |
| `empty_page_text` | 5 | `outputs/badcases/ingest_badcases.jsonl` |
| `empty_after_cleaning` | 5 | `outputs/badcases/ingest_badcases.jsonl` |

| Chunk 策略 | `avg_chunk_count` | `avg_char_count` | `avg_table_like_ratio` | 证据来源 |
|---|---:|---:|---:|---|
| `fixed_window` | 25.78 | 665.49 | 0.00 | `data/reports/chunk_strategy_report.json` |
| `title_aware` | 43.43 | 295.90 | 0.00 | `data/reports/chunk_strategy_report.json` |
| `table_protected` | 78.50 | 193.45 | 0.28 | `data/reports/chunk_strategy_report.json` |

### A.2 Retrieval

| 方法 | `Recall@5` | `Recall@10` | `MRR@10` | 证据来源 |
|---|---:|---:|---:|---|
| Dense | 0.9444 | 1.0000 | 0.6431 | `outputs/reports/retrieval_eval_summary.json` |
| BM25 | 0.7222 | 0.8889 | 0.5667 | `outputs/reports/retrieval_eval_summary.json` |
| Hybrid | 0.8889 | 1.0000 | 0.6519 | `outputs/reports/retrieval_eval_summary.json` |
| Hybrid+Rerank | 1.0000 | 1.0000 | 0.9167 | `outputs/reports/retrieval_eval_summary.json` |

| 指标 | 数值 | 证据来源 |
|---|---:|---|
| retrieval badcase 数量 | 2 | `outputs/badcases/retrieval_badcases.jsonl` |

### A.3 Answer / Abstain / Eval

| 阶段 | `Answer Hit Rate` | `Citation Hit Rate` | `Abstain Precision` | `Abstain Recall` | `Unsupported Answer Count` |
|---|---:|---:|---:|---:|---:|
| dev 第一次 | 0.1667 | 0.1667 | 0.0000 | 1.0000 | 0 |
| dev 第二次 | 0.2778 | 0.3333 | 0.0000 | 1.0000 | 0 |
| dev 最终 | 0.5556 | 0.5556 | 1.0000 | 1.0000 | 0 |
| full 当前 | 0.3111 | 0.3556 | 0.0667 | 0.2000 | 0 |

> 说明：dev 第一次、第二次指标来自阶段三调参过程记录；dev 最终和 full 当前来自远程产物：
> - `outputs/reports/answer_eval_summary_dev.json`
> - `outputs/reports/answer_eval_summary_full.json`

| 指标 | 数值 | 证据来源 |
|---|---:|---|
| `answer_badcases_dev` | 8 | `outputs/badcases/answer_badcases_dev.jsonl` |
| `answer_badcases_full` | 31 | `outputs/badcases/answer_badcases_full.jsonl` |
| `abstain_badcases_full` | 4 | `outputs/badcases/abstain_badcases_full.jsonl` |

### A.4 Full 集延迟

| 指标 | 数值 | 证据来源 |
|---|---:|---|
| `Avg Retrieval Latency` | 30.84 ms | `outputs/reports/latency_summary_full.json` |
| `Avg Rerank Latency` | 131.48 ms | `outputs/reports/latency_summary_full.json` |
| `Avg Generation Latency` | 1604.68 ms | `outputs/reports/latency_summary_full.json` |
| `End-to-End Latency` | 1772.48 ms | `outputs/reports/latency_summary_full.json` |

## B. badcase 统计

### B.1 Ingest badcase 分布

| `issue_type` | 数量 |
|---|---:|
| `empty_page_text` | 5 |
| `empty_after_cleaning` | 5 |

### B.2 Answer badcase 分布

| 文件 | 总数 | fact | comparison | inductive | 说明 |
|---|---:|---:|---:|---:|---|
| `answer_badcases_dev.jsonl` | 8 | 3 | 1 | 4 | dev 集复杂题失败比例已经偏高 |
| `answer_badcases_full.jsonl` | 31 | 9 | 8 | 14 | full 集中 inductive 失败最多 |

### B.3 Abstain badcase

| 文件 | 数量 | 含义 |
|---|---:|---|
| `abstain_badcases_full.jsonl` | 4 | full 集 must_abstain 样本中，仍有 4 条未正确拒答 |

### B.4 retrieval badcase 示例

```json
{"question_id":"fact_05","question_type":"fact","method":"bm25","query":"哪份农林牧渔周报写到猪价持续低迷、重视板块去产能投资机会？"}
```

现象说明：

- gold 文档是国金证券周报
- BM25 top5 主要召回山西证券、东兴证券等相似农业周报
- 说明在高模板化行业周报里，词面检索容易命中“相似但不正确”的文档

### B.5 answer badcase 示例

#### 示例 1：事实题答案正确，但 citation gold 粒度不一致

```json
{"question_id":"fact_04","answer_hit":false,"citation_hit":false,"final_answer":"东兴证券发布的《农林牧渔行业生猪养殖行业月度跟踪：节后猪价加速下行，产能去化在即》报告强调了节后猪价加速下行、产能去化在即。"}
```

现象说明：

- 语言答案本身是正确的
- 但模型引用到了第 `3` 页和第 `9` 页的块
- gold 标注只保留了首页代表性 chunk，导致 citation 和 answer hit 同时被拉低

#### 示例 2：事实题证据命中了正文块而不是标题块

```json
{"question_id":"fact_05","answer_hit":false,"citation_hit":false,"final_answer":"国金证券发布的农林牧渔行业周报《猪价持续低迷，重视板块去产能投资机会》提到猪价持续低迷，强调了重视板块去产能投资机会。"}
```

现象说明：

- 回答内容正确
- 但模型最终引用的是正文页块，而 gold 是首页标题 chunk
- 暴露出“答案正确”与“citation gold 粒度过细”之间的结构性矛盾

## C. 关键修复前后对比

### C.1 dev 指标收敛过程

| 轮次 | 主要修复动作 | `Answer Hit Rate` | `Citation Hit Rate` | `abstained_count` |
|---|---|---:|---:|---:|
| 初版 | 刚接入回答层 | 0.1667 | 0.1667 | 9 |
| 第二版 | 放宽 support validation | 0.2778 | 0.3333 | 6 |
| 最终版 | 修正 evidence-id 数字误判与 citation 粒度 | 0.5556 | 0.5556 | 0 |

### C.2 检索层关键对比

| 对比项 | 修复前 / 旧方案 | 修复后 / 现方案 |
|---|---|---|
| 最佳检索排序 | Dense / Hybrid 排序不稳 | Hybrid+Rerank 成为默认生产入口 |
| `MRR@10` | Hybrid `0.6519` | Hybrid+Rerank `0.9167` |
| retrieval badcase | BM25 仍有误召回 | 默认生产链路 badcase 降到 0 |

### C.3 事实题评测误判修复

| 问题 | 修复前 | 修复后 |
|---|---|---|
| `short_title()` 过严 | `fact_01` 单题 answer hit = 0 | 单题 smoke test answer hit = 1 |
| paraphrase 被误判 | 多个事实题被拒答为 `unsupported_answer` | 只保留高置信规则，dev 恢复到 0 个误拒答 |
| `E1/E2/E3` 被当数字 | `compare_04` 触发 `unsupported_numeric_claim` | 过滤 evidence id 后误拒答消失 |

## D. 关键命令现象摘录

### D.1 PowerShell 引号嵌套失败

```text
At line:2 char:576
An expression was expected after '('.
```

### D.2 远程未激活 `.venv`

```text
bash: line 1: python: command not found
```

### D.3 Windows 本地输出编码问题

```text
UnicodeEncodeError: 'gbk' codec can't encode character '\uf06e'
```

### D.4 首次模型下载

```text
Downloading Model from https://www.modelscope.cn to directory: /root/autodl-tmp/financial-rag-project/models/Qwen/Qwen2.5-7B-Instruct
```

### D.5 仍然存在的生成警告

```text
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k'].
```

说明：

- 该告警在阶段三后期仍可见
- 这说明回答链路虽然可运行，但生成配置和模型默认 `generation_config` 仍未完全清理

## E. 产物文件索引

以下文件是本轮复盘最关键的证据来源，均位于远程项目目录 `/root/autodl-tmp/financial-rag-project/`：

| 文件 | 用途 |
|---|---|
| `data/reports/ingest_summary.json` | Ingest 总量统计 |
| `data/reports/chunk_strategy_report.json` | Chunk 策略对比结果 |
| `outputs/reports/retrieval_eval_summary.json` | 检索阶段指标汇总 |
| `outputs/reports/answer_eval_summary_dev.json` | dev 答案级评测结果 |
| `outputs/reports/answer_eval_summary_full.json` | full 答案级评测结果 |
| `outputs/reports/latency_summary_full.json` | full 集延迟统计 |
| `outputs/badcases/ingest_badcases.jsonl` | Ingest 问题样本 |
| `outputs/badcases/retrieval_badcases.jsonl` | retrieval 失败样本 |
| `outputs/badcases/answer_badcases_dev.jsonl` | dev 回答失败样本 |
| `outputs/badcases/answer_badcases_full.jsonl` | full 回答失败样本 |
| `outputs/badcases/abstain_badcases_full.jsonl` | full 拒答失败样本 |

补充本地快照：

| 文件 | 用途 |
|---|---|
| `tmp_phase2/retrieval_eval_summary.json` | 阶段二本地检索指标快照 |
| `data/eval_set/retrieval_eval_seed.jsonl` | retrieval eval seed 数据 |
| `data/eval_set/retrieval_eval.jsonl` | retrieval eval 物化数据 |
