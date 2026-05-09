# 单-双折叠翼变体飞行器知识图谱

基于论文 `aftcln.txt`（清华大学郭廷宇博士论文）构建领域知识图谱：`ans.py` 提供人工整理领域词典，关系由多算法自动抽取；另含银标评估与简易可视化应用。

## 环境

```bash
uv sync
# 中文 spaCy 模型（用于 SVO 依存抽取）
uv run python -m spacy download zh_core_web_sm
# 可选：Streamlit 浏览应用
uv sync --extra app
```

## 常用命令

| 说明 | 命令 |
|------|------|
| 自动抽取（默认 mock LLM，无需 API） | `uv run python run_extract.py` |
| **仅用 JSON 词表、运行时不读 ans** | 先 `uv run python tools/export_entities_by_type_json.py`，再 `uv run python run_extract.py --entities-json data/entities_by_type.json` |
| 查看领域词典条目数 | `uv run python ans.py` 或 `uv run python main.py manual` |
| 导出银标（≥400 三元组、≥200 实体量级） | `uv run python tools/export_silver_gold.py` |
| 与银标对比算 P/R/F1 | `uv run python evaluate_kg.py` |
| 图谱浏览（需已安装 app 依赖） | `uv run streamlit run app_streamlit.py` |

统一入口（子命令转发）：

```bash
uv run python main.py extract --help
uv run python main.py manual
uv run python main.py silver
uv run python main.py eval
uv run python main.py app
```

## 输出文件

- `output/triples_with_meta.csv`：带 `source` / `score` / `chapter` / `sentence` 的三元组
- `output/entities.csv`、`output/extraction_stats.txt`（含 **论文实体挖掘** 统计：候选串数、新并入词典词数）
- `knowledge_graph.csv`：简版三列（兼容旧格式；**自动抽取**写入）
- `gold/silver_entities.txt`、`gold/silver_triples.csv`：银标说明见 `gold/README_SILVER.txt`
- `output/eval_report.txt`：`evaluate_kg.py` 的指标摘要

## 论文实体挖掘（默认开启）

在触发词 / 共现 / SVO 等抽取**之前**，用 **spaCy**（`zh_core_web_sm`）对每句做命名实体、`noun_chunks`、连续名/专名拼接，从正文里挖出**原领域词表未收录但文中确有**的字符串，经停用词与长度过滤后，通过 `HybridNER.add_terms` 并入 AC 自动机，再参与后续三元组抽取。这样「先扩真实实体，再成边」，有利于提高 `output/entities.csv` 覆盖。

```bash
# 关闭挖掘，仅用 ans / JSON 原词表
uv run python run_extract.py --no-paper-entity-mine

# 挖掘串在全文至少出现 2 次才入库（降噪）
uv run python run_extract.py --paper-entity-min-freq 2
```

## LLM 增强（可选）

```bash
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set OPENAI_MODEL=deepseek-chat
uv run python run_extract.py --llm openai --entities-json data/entities_by_type.json
```

二阶段「发现补全」（**按章**把聚合正文 + 已有候选发给模型，允许补充**正文中字面出现**的新头/尾实体；代码侧会校验 `head`/`tail` 必须是正文子串，`relation` 必须在允许集合内）：

```bash
uv run python run_extract.py --llm openai --llm-discover --entities-json data/entities_by_type.json
# 可调：每章最多采纳条数、传给模型的候选行数
uv run python run_extract.py --llm openai --llm-discover
# 显式改小/改大（默认每章最多 40 条新边）：
uv run python run_extract.py --llm openai --llm-discover --llm-discover-max-new 25 --llm-discover-max-lines 80
```

发现阶段发给模型的正文长度默认 **16000** 字（与 `PipelineConfig.llm_discovery_max_chars` 一致），可用环境变量 `KG_LLM_DISCOVER_CHARS` 调整。

首轮 LLM 质检按批送候选，默认每批 **50** 条，可用 **`KG_LLM_MAX_BATCH`**（整数）调整，避免单请求过大。

首轮送入提示的**按章正文**长度默认 **16000** 字（`pipeline` 拼接与 `_llm_enhance` 截断一致），可用 **`KG_LLM_POLISH_CHARS`** 调大/调小；仍长于该值的章会被截断，超长需分窗或多轮（未自动实现）。

也可用环境变量 `KG_ENTITIES_JSON` 指向 JSON，省略每次写 `--entities-json`。

## 银标与课程要求

银标由「人工词表 + 关系生成器」产生、且头尾实体均在论文正文中出现，用于**可复现**的自动评估；若课程要求**逐句人工金标**，请以 `gold/silver_triples.csv` 为模板另存为 `gold/gold_triples_labeled.csv` 并人工修订后，用 `evaluate_kg.py --gold` 指向该文件。
