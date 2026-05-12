# 项目构建问题复盘

## 1. 项目背景与构建范围

本项目的目标是基于金融研究 PDF 搭建一个可落地的 RAG 系统，覆盖三条主线：

- 阶段一：PDF 解析、清洗、Chunk 与中间产物落盘
- 阶段二：Dense / BM25 / Hybrid / Rerank 检索基线
- 阶段三：本地开源答案生成、引用绑定、规则式拒答与答案级评测

本轮构建不是在本地机器完成，而是迁移到远程服务器 `/root/autodl-tmp/financial-rag-project` 上进行开发和验证。项目使用的数据规模和当前稳定产物如下：

| 项目 | 数值 |
|---|---:|
| PDF 数量 | 54 |
| 页数 | 680 |
| 生产 Chunk 数量 | 4239 |
| Ingest badcase | 10 |
| retrieval eval 题量 | 18 |
| answer eval dev 题量 | 18 |
| answer eval full 题量 | 50 |

本复盘文档聚焦“构建过程中遇到的问题”，而不是功能说明。所有结论优先使用现有 JSON/JSONL 产物、远程评测结果和实际执行现象作为证据。

## 2. 构建时间线

| 时间 | 阶段 | 关键动作 | 结果 |
|---|---|---|---|
| 2026-03-31 | 环境切换 | 由于本地无法实施，切换到远程服务器开发 | 远程目录和 Python 环境建立完成 |
| 2026-03-31 | 阶段一 | 完成 PDF 解析、清洗、Chunk 产物落盘 | 得到 `parsed_pages`、`cleaned_pages`、`chunks` |
| 2026-03-31 | 阶段二 | 完成 Dense / BM25 / Hybrid / Rerank 检索基线 | retrieval 指标收敛，Hybrid+Rerank 最优 |
| 2026-03-31 | 阶段三初版 | 新增 `generation/`、`run_answer_eval.py`、answer eval 数据集物化 | smoke test 跑通，但答案级指标较差 |
| 2026-03-31 | 阶段三调参 1 | 修正 `short_title()` 事实题评测逻辑 | 单题 smoke test 从误判恢复为正确命中 |
| 2026-03-31 | 阶段三调参 2 | 放宽支持性校验，避免把合理 paraphrase 误判成 `unsupported_answer` | dev 指标从 `0.1667 / 0.1667` 提升到 `0.2778 / 0.3333` |
| 2026-03-31 | 阶段三调参 3 | 修正 evidence id 被误判为数字 claim、调整 citation 命中文档粒度 | dev 指标提升到 `0.5556 / 0.5556` |
| 2026-03-31 | 阶段三最终验证 | 跑完 dev 18 条与 full 50 条 | dev 可用，full 仍明显不足 |

## 3. 环境与远程开发问题

### 问题 ENV-01：远程服务器规格与预期不一致

**问题编号**：ENV-01  
**问题现象**：最初按“租用 4090 服务器”规划显存与运行时，但实际远程 GPU 不是 4090。  
**出现阶段**：远程环境初始化  
**证据**：远程 `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` 返回 `NVIDIA GeForce RTX 5090, 32607 MiB`。  
**根因判断**：租用信息与实际实例规格存在偏差，前期方案默认假设了 4090。  
**解决动作**：后续所有生成与评测默认按“本地单卡 32GB 级显存”重新估算，不再按 24GB 规格保守设计。  
**修复前后对比**：`预期设备=RTX 4090` -> `实际设备=RTX 5090 32GB`  
**当前状态**：已确认，已适配

### 问题 ENV-02：PowerShell 引号嵌套导致远程探测命令失败

**问题编号**：ENV-02  
**问题现象**：通过本地 PowerShell 包裹远程命令时，带有多层引号和 Python one-liner 的命令直接解析失败。  
**出现阶段**：远程依赖探测  
**证据**：命令报错：`At line:2 char:576 ... An expression was expected after '('`。  
**根因判断**：Windows PowerShell、SSH 远程命令和 Python 内联字符串同时存在转义层，导致引号嵌套失控。  
**解决动作**：将依赖探测改成更稳的 `pip show` / `python -V` 组合，复杂统计改为 here-doc 形式，并减少 PowerShell 内联表达式层级。  
**修复前后对比**：`内联 Python 失败` -> `pip show / here-doc 命令可执行`  
**当前状态**：已解决

### 问题 ENV-03：远程命令未激活 `.venv` 导致 `python: command not found`

**问题编号**：ENV-03  
**问题现象**：部分远程统计命令直接调用 `python` 时失败。  
**出现阶段**：远程 badcase 统计与产物核验  
**证据**：错误信息：`bash: line 1: python: command not found`。  
**根因判断**：远程机器默认 shell 环境中未自动激活项目 `.venv`，命令假设不成立。  
**解决动作**：所有远程 Python 相关命令统一前置 `. .venv/bin/activate`。  
**修复前后对比**：`python: command not found` -> `.venv` 激活后统计命令正常执行  
**当前状态**：已解决

### 问题 ENV-04：Windows 本地终端读取远程 UTF-8 输出时触发 `UnicodeEncodeError`

**问题编号**：ENV-04  
**问题现象**：远程 JSON/结果里包含 UTF-8 字符，本地 Windows 控制台按 GBK 输出时直接失败。  
**出现阶段**：读取远程 answer eval 结果  
**证据**：错误信息：`UnicodeEncodeError: 'gbk' codec can't encode character '\uf06e'`。  
**根因判断**：本地控制台编码与远程 UTF-8 文本不兼容，属于结果读取链路问题，不是远程执行问题。  
**解决动作**：本地执行远程命令前显式设置 `PYTHONIOENCODING=utf-8`。  
**修复前后对比**：`无法打印远程 UTF-8 结果` -> `UTF-8 模式下可读取 answer eval JSON`  
**当前状态**：已解决

### 问题 ENV-05：首次远程 smoke test 触发大模型下载，导致运行时长异常

**问题编号**：ENV-05  
**问题现象**：第一次只跑 `1` 条 dev 样本，但整次 smoke test 耗时接近 15 分钟。  
**出现阶段**：阶段三首次 smoke test  
**证据**：

- 首次远程执行前打印：`Downloading Model from https://www.modelscope.cn to directory: /root/autodl-tmp/financial-rag-project/models/Qwen/Qwen2.5-7B-Instruct`
- 随后开始下载 `4` 个 `safetensors` 分片，每个分片约 `3.3GB-3.7GB`

**根因判断**：首次运行时，`Qwen/Qwen2.5-7B-Instruct` 尚未缓存，本次 smoke test 同时承担了模型下载和推理验证。  
**解决动作**：后续所有 smoke test、dev/full 评测都复用缓存后的本地模型目录。  
**修复前后对比**：`首次单题 smoke test` 明显异常慢 -> 缓存完成后 `dev 18 条` 在约 `52s` 内跑完  
**当前状态**：已解决

## 4. 数据入库与 Chunk 问题

### Ingest / Chunk 基线数据

| 指标 | 数值 |
|---|---:|
| PDF 数量 | 54 |
| 页数 | 680 |
| 生产 Chunk 数量 | 4239 |
| Ingest badcase | 10 |
| `empty_page_text` | 5 |
| `empty_after_cleaning` | 5 |

| Chunk 策略 | `avg_chunk_count` | `avg_char_count` | `avg_table_like_ratio` |
|---|---:|---:|---:|
| `fixed_window` | 25.78 | 665.49 | 0.00 |
| `title_aware` | 43.43 | 295.90 | 0.00 |
| `table_protected` | 78.50 | 193.45 | 0.28 |

### 问题 DATA-01：PDF 解析链路存在空页与清洗后空文本页

**问题编号**：DATA-01  
**问题现象**：部分 PDF 页在解析阶段提取不到文本，或者清洗后被全部删除。  
**出现阶段**：阶段一 ingest  
**证据**：`outputs/badcases/ingest_badcases.jsonl` 统计为：

- `empty_page_text = 5`
- `empty_after_cleaning = 5`

**根因判断**：

- 部分页面本身就是扫描/图片页或版式极弱页
- 部分页面文本提取后主要是页眉、页脚、网页导航、空白噪声，清洗后全部消失

**解决动作**：保留 badcase 记录，不在首版强行 OCR；清洗环节只做噪声剔除与结构保留，不引入复杂修复。  
**修复前后对比**：`无问题定位能力` -> `所有空页/清洗后空页都有 issue_type 记录`  
**当前状态**：部分解决，已可追踪但未彻底修复

### 问题 DATA-02：东方财富网页导出 PDF 噪声难以完全去除

**问题编号**：DATA-02  
**问题现象**：网页导出类 PDF 存在导航条、站点页脚、风格化信息块等噪声，影响 Chunk 纯度。  
**出现阶段**：阶段一 ingest，阶段三 answer 阶段放大暴露  
**证据**：

- `source_type` 已区分 `web_export_pdf`
- answer badcase 样本中出现与正文无关的“相关报告汇总”类片段进入证据，例如 `fact_04` 的 `E2` 引到了第 `9` 页“相关报告汇总”

**根因判断**：网页导出 PDF 的结构并非传统研报排版，纯规则清洗无法稳定剔除所有噪声块。  
**解决动作**：当前通过 `guess_source_type()` 标记来源类型，靠清洗与标题感知/表格保护先做粗过滤。  
**修复前后对比**：`网页导出 PDF 与普通研报混处理` -> `已可识别 source_type，但噪声仍会进入下游`  
**当前状态**：未完全解决

### 问题 DATA-03：Chunk 策略之间存在明显权衡，生产策略不是“越大越好”

**问题编号**：DATA-03  
**问题现象**：不同 Chunk 策略在信息密度、表格保留和下游召回表现之间差异非常大。  
**出现阶段**：阶段二前置数据整理  
**证据**：

- `fixed_window`：Chunk 少，但平均长度 `665.49`
- `title_aware`：更细，但表格识别不足
- `table_protected`：平均长度只有 `193.45`，Chunk 数上升到 `78.5 / doc`，同时 `table_like_ratio = 0.28`

**根因判断**：

- 大块文本利于语义完整性，但会把表格、标题和正文混在一起
- 金融 PDF 中大量问题与数字、标题边界、表格结构相关，纯固定窗口不够稳

**解决动作**：选择 `table_protected` 作为生产策略，并保留三套实验产物作为对照。  
**修复前后对比**：`只有 baseline chunking` -> `三策略可对比，生产策略固定为 table_protected`  
**当前状态**：已解决，但代价是 Chunk 数量显著上升

## 5. 检索基线问题

### Retrieval 基线数据

| 方法 | `Recall@5` | `Recall@10` | `MRR@10` |
|---|---:|---:|---:|
| Dense | 0.9444 | 1.0000 | 0.6431 |
| BM25 | 0.7222 | 0.8889 | 0.5667 |
| Hybrid | 0.8889 | 1.0000 | 0.6519 |
| Hybrid+Rerank | 1.0000 | 1.0000 | 0.9167 |

retrieval badcase 总数为 `2`。

### 问题 RET-01：BM25 在高词面相似场景下容易召回“相似但不正确”的报告

**问题编号**：RET-01  
**问题现象**：BM25 在同类农业/生猪主题周报上容易把近义词和相似模板标题混在一起，命中错误文档。  
**出现阶段**：阶段二 retrieval eval  
**证据**：`outputs/badcases/retrieval_badcases.jsonl` 中：

- `fact_05` 真实目标是国金周报“猪价持续低迷，重视板块去产能投资机会”
- BM25 top5 却主要召回山西证券和东兴证券等相似周报

**根因判断**：农业行业周报标题与正文高度模板化，BM25 更偏向词面重合，难以区分“去产能”“猪价低迷”“回归和演绎”等近似叙述。  
**解决动作**：保留 BM25 作为 lexical baseline，但默认生产检索链路改为 Hybrid+Rerank。  
**修复前后对比**：`BM25 仍有 2 条 badcase` -> `Hybrid+Rerank 在 18 条 retrieval eval 上无 badcase`  
**当前状态**：已解决为“基线问题”，未继续单独优化 BM25

### 问题 RET-02：仅靠 Dense 或 Hybrid 不足以稳定排序，需要 Rerank 做最后一跳

**问题编号**：RET-02  
**问题现象**：Dense 与 Hybrid 能把正确文档召回，但排序不够稳定，尤其在多份相似报告并存时，正确证据不一定排在最前。  
**出现阶段**：阶段二 retrieval eval  
**证据**：

- Dense 的 `MRR@10 = 0.6431`
- Hybrid 的 `MRR@10 = 0.6519`
- Hybrid+Rerank 的 `MRR@10 = 0.9167`

**根因判断**：召回层解决的是“找回来”，不是“排最准”；金融 PDF 查询里标题、正文、术语和表格数字混合，最终排序需要交叉编码器校准。  
**解决动作**：在 Hybrid 结果上接 `BAAI/bge-reranker-v2-m3`，默认生产入口使用 `Hybrid+Rerank Top5`。  
**修复前后对比**：`Hybrid MRR@10 0.6519` -> `Hybrid+Rerank MRR@10 0.9167`  
**当前状态**：已解决

## 6. 生成、引用绑定与拒答问题

### Answer / Abstain / Eval 基线数据

| 阶段 | `Answer Hit Rate` | `Citation Hit Rate` | `Abstain Precision` | `Abstain Recall` |
|---|---:|---:|---:|---:|
| dev 第一次 | 0.1667 | 0.1667 | 0.0000 | 1.0000 |
| dev 第二次 | 0.2778 | 0.3333 | 0.0000 | 1.0000 |
| dev 最终 | 0.5556 | 0.5556 | 1.0000 | 1.0000 |
| full 当前 | 0.3111 | 0.3556 | 0.0667 | 0.2000 |

补充数据：

- `answer_badcases_dev = 8`
- `answer_badcases_full = 31`
- `abstain_badcases_full = 4`
- `Unsupported Answer Count = 0`
- `full` 延迟：
  - `Avg Retrieval Latency = 30.84 ms`
  - `Avg Rerank Latency = 131.48 ms`
  - `Avg Generation Latency = 1604.68 ms`
  - `End-to-End Latency = 1772.48 ms`

### 问题 GEN-01：事实题明明答对，但 `short_title()` 解析逻辑过严导致评测误判

**问题编号**：GEN-01  
**问题现象**：单题 smoke test 中，模型输出了正确报告名，但评测仍判定 `Answer Hit Rate = 0`。  
**出现阶段**：阶段三第一次 smoke test  
**证据**：

- 事实题 `fact_01` 的模型回答明确指向正确报告
- 但第一次单题 summary 为：`Answer Hit Rate = 0.0`，`Citation Hit Rate = 1.0`

**根因判断**：评测函数中 `short_title()` 对中文标题、日期后缀和冒号分割处理过于死板，导致正确答案被错误裁剪。  
**解决动作**：修正标题截断逻辑，优先按中文标题语义截取，再单独去掉日期尾缀。  
**修复前后对比**：`fact_01: Answer Hit Rate 0.0 -> 1.0`  
**当前状态**：已解决

### 问题 GEN-02：支持性校验过严，把合理 paraphrase 误判成 `unsupported_answer`

**问题编号**：GEN-02  
**问题现象**：模型答案与证据方向一致，但只因为措辞不是逐词复述，就被规则拒答。  
**出现阶段**：阶段三 dev 初版  
**证据**：

- 第一版 dev 结果中 `abstained_count = 9`
- 当时多个事实题被标记为 `unsupported_answer`
- 例如 `fact_03`、`fact_05`、`fact_06`、`fact_07` 的 `missing_keywords` 中出现了大量“发布、年度、策略、证券”等总结性表述

**根因判断**：支持性校验最初按“回答词项是否能在证据中逐个找到”来做，惩罚了合理概括与 paraphrase。  
**解决动作**：校验逻辑改为“查询关键词 + 真实数字是否有支撑”，不再拿回答里的所有表述逐词对齐证据。  
**修复前后对比**：

- `dev Answer Hit Rate: 0.1667 -> 0.2778 -> 0.5556`
- `dev Citation Hit Rate: 0.1667 -> 0.3333 -> 0.5556`
- `dev abstained_count: 9 -> 6 -> 0`

**当前状态**：已解决为可用版本，但 full 集仍表现不足

### 问题 GEN-03：`E1/E2/E3` 被误当成数字 claim，触发 `unsupported_numeric_claim`

**问题编号**：GEN-03  
**问题现象**：比较题和归纳题的 `evidence_summary` 中会提到 `E1/E2/E3`，校验器把其中的数字当成真实数值去匹配证据。  
**出现阶段**：阶段三中期调参  
**证据**：

- `compare_04` 一度被拒答，原因为 `unsupported_numeric_claim`
- 当时校验器识别出的 `missing_numeric_tokens` 包含 `1`、`3`、`4`

**根因判断**：数字抽取器没有区分“证据编号”和“财务/年份/百分比”这类真实 claim。  
**解决动作**：先剔除 `E\d+` 形式，再做数字 claim 提取；同时过滤纯日期尾缀。  
**修复前后对比**：`compare_04` 从误拒答恢复为可生成，`dev abstained_count` 最终降到 `0`  
**当前状态**：已解决

### 问题 GEN-04：comparison / inductive 的 citation 命中按 chunk/page 判太严

**问题编号**：GEN-04  
**问题现象**：模型已经引用了正确文档，但因为 gold 标注只保留“代表性 chunk”，导致 citation 命中率被低估。  
**出现阶段**：阶段三中期调参  
**证据**：

- `compare_02`、`compare_03`、`compare_06` 等题中，模型已引用正确的两个文档
- 但早期逻辑按 chunk/page 对齐，comparison / inductive 题经常被判 `citation_hit = false`

**根因判断**：comparison / inductive 的 gold 标注本质上是“文档级/主题级”证据，而非单一 chunk 精确定位。  
**解决动作**：对 comparison / inductive 将 citation 命中规则改为“文档级命中为主，chunk/page overlap 为辅”。  
**修复前后对比**：`dev Citation Hit Rate 0.3333 -> 0.5556`  
**当前状态**：已解决为更合理的评测口径

### 问题 GEN-05：生成端为了压制幻觉，把 `Unsupported Answer Count` 压到了 0，但牺牲了 full 集效果

**问题编号**：GEN-05  
**问题现象**：从安全性看，系统已经能做到“unsupported answer 不外泄”；但从效果看，比较题和归纳题仍明显偏弱。  
**出现阶段**：阶段三最终验证  
**证据**：

- dev：`Unsupported Answer Count = 0`
- full：`Unsupported Answer Count = 0`
- 但 full：`Answer Hit Rate = 0.3111`，`Citation Hit Rate = 0.3556`

**根因判断**：当前规则式校验非常强调“证据充分性”，对事实题更友好，但对 comparison / inductive 的多证据融合和抽象表达支持不足。  
**解决动作**：本轮只做规则收敛，不继续扩复杂生成器或 judge。  
**修复前后对比**：`unsupported answer 风险被控制`，但 `full` 效果仍未达标  
**当前状态**：部分解决

## 7. 评测设计与数据集问题

### 问题 EVAL-01：dev 与 full 指标存在明显落差，说明自动扩展评测集质量不足

**问题编号**：EVAL-01  
**问题现象**：dev 集效果明显高于 full 集。  
**出现阶段**：阶段三批量评测  
**证据**：

- dev：`Answer Hit Rate 0.5556`，`Citation Hit Rate 0.5556`
- full：`Answer Hit Rate 0.3111`，`Citation Hit Rate 0.3556`

**根因判断**：dev 18 条来自 retrieval eval 的人工 seed，语义更稳定；full 50 条中有相当一部分是自动生成题，gold answer 和 must_abstain 标注质量不如 dev。  
**解决动作**：当前保留 full 作为“压力测试集”，但不把它当成人工金标准。  
**修复前后对比**：`dev 18 条人工种子集` -> `full 50 条自动扩展集`，覆盖面扩大但质量下降  
**当前状态**：未解决

### 问题 EVAL-02：full 集拒答质量很差，说明负样本设计还不成熟

**问题编号**：EVAL-02  
**问题现象**：系统在 full 集上的拒答精度很低。  
**出现阶段**：阶段三 full 评测  
**证据**：

- full：`must_abstain_query_count = 5`
- full：`Abstain Precision = 0.0667`
- full：`Abstain Recall = 0.2`

这意味着：

- 5 条应拒答样本里，只正确拒答了 1 条
- 系统总共大约做出了 15 次拒答，其中绝大多数是误拒答

**根因判断**：当前负样本来自自动构造的跨领域混搭问题，形式上像“应拒答”，但与真实用户问题分布并不一致；同时规则式拒答对长问题和组合问题敏感。  
**解决动作**：本轮只记录问题，不继续人工重标 full 集。  
**修复前后对比**：无  
**当前状态**：未解决

### 问题 EVAL-03：comparison / inductive 题型显著难于 fact

**问题编号**：EVAL-03  
**问题现象**：复杂题型在 full 集上的 badcase 占比明显更高。  
**出现阶段**：阶段三 full 评测  
**证据**：`outputs/badcases/answer_badcases_full.jsonl` 分布如下：

- `fact = 9`
- `comparison = 8`
- `inductive = 14`

在总题量分别为 `20 / 15 / 15` 的前提下，inductive 的失败密度最高。  
**根因判断**：comparison / inductive 需要多证据选择、跨文档融合和更强的抽象总结能力，当前规则式 evidence selection 与 prompt 主要针对 fact 做了优化。  
**解决动作**：当前仅通过“至少保留两个来源”“文档级 citation 命中”降低明显误判。  
**修复前后对比**：`fact` 类问题基本可用，`comparison / inductive` 仍显著偏弱  
**当前状态**：未解决

## 8. 当前未解决问题

### 8.1 full 集答案质量仍明显偏低

- **当前表现**：`Answer Hit Rate = 0.3111`
- **业务影响**：系统在大规模复杂问题上仍不具备稳定回答能力，不能据此宣称“回答层已达到生产可用”
- **为什么没解决**：当前只做了规则收敛，没有继续优化 comparison / inductive 的 prompt、evidence selection 和 answer aggregation
- **下一阶段怎么处理**：优先对 comparison / inductive 单独设计 evidence routing 和 prompt 模板

### 8.2 full 集 citation 命中仍偏低

- **当前表现**：`Citation Hit Rate = 0.3556`
- **业务影响**：即使答案大方向正确，也可能无法稳定给出与 gold 标注一致的引用
- **为什么没解决**：当前 gold 与模型引用的粒度仍存在差异，尤其是主题型问题
- **下一阶段怎么处理**：补人工 citation 标注，区分“文档级命中”和“chunk 级命中”

### 8.3 full 集拒答精度过低

- **当前表现**：`Abstain Precision = 0.0667`
- **业务影响**：系统在 full 集上大量误拒答，说明规则版拒答无法直接用于真实复杂问题
- **为什么没解决**：must_abstain 负样本设计粗糙，拒答规则对复杂语义结构不够鲁棒
- **下一阶段怎么处理**：重做负样本集，并引入更强的判别式支持校验

### 8.4 自动生成的 full eval 集本身可能存在标注质量问题

- **当前表现**：full 集由 `18` 条 seed 扩展到 `50` 条，其中包含自动生成的 fact / comparison / inductive 与 `5` 条 must_abstain
- **业务影响**：当前 full 指标更适合作为“压力测试”而不是最终答辩口径
- **为什么没解决**：本轮优先验证链路可运行，没有对 full 全量题做人工审校
- **下一阶段怎么处理**：人工复核 full 50 条题，重点复核 `gold_answer`、`gold_chunk_ids` 和 `must_abstain`

### 8.5 comparison / inductive 题型仍显著弱于 fact

- **当前表现**：`answer_badcases_full` 中 `inductive = 14`，为所有题型最高
- **业务影响**：对用户最有价值的总结型问题，目前恰恰是最不稳定的
- **为什么没解决**：当前证据选择和 prompt 仍偏向“报告标题识别型”事实题
- **下一阶段怎么处理**：增加多证据聚合、主题句提取、表格证据加权和分题型 prompt

### 8.6 规则版拒答仍是过渡方案

- **当前表现**：dev 上可用，full 上明显失效
- **业务影响**：规则能防幻觉，但无法提供稳定的复杂问题风险控制
- **为什么没解决**：本轮没有引入学习式/判别式 verifier，也没有引入 LLM judge
- **下一阶段怎么处理**：将拒答与支持性校验从纯规则升级为“规则 + 判别模型”组合

## 9. 结论与下一步建议

本轮项目构建在工程闭环上已经完成了三个关键目标：

- 数据链路打通：`54` 份 PDF 已完成解析、清洗、Chunk 与落盘
- 检索链路收敛：`Hybrid+Rerank` 在 retrieval eval 上明显优于其他基线
- 回答链路跑通：本地开源生成、引用绑定、拒答和答案级评测均可在远程服务器上完整执行

但从复盘结果看，当前真正“已经解决”的主要是工程问题和检索问题；真正“还没有解决”的，是比较题、归纳题和高质量 full 评测集上的答案质量问题。

下一阶段建议按以下顺序推进：

1. 先人工审校 full 50 条评测集，不再把当前 full 指标当作最终口径  
2. 对 comparison / inductive 单独设计 evidence selection 与 prompt  
3. 重新设计 must_abstain 负样本，并升级拒答逻辑  
4. 在 answer 层稳定后，再考虑 Demo、在线问答和更复杂的 judge / verifier
