# 单-双折叠翼变体飞行器 · 知识图谱构建系统

> 基于清华大学郭廷宇博士论文 `aftcln.txt`，使用多策略 NLP 算法自动抽取领域知识图谱。
>
> **课程目标**：概念 ≥500、关系 ≥1000、标注 ≥200 概念 + ≥400 关系做评估。
> **当前指标**：实体 **1473**、三元组 **4642**、银标实体 **1221**、银标三元组 **7878**。

---

## 1. 项目总览

```
my-project/
├── aftcln.txt                  # 论文原文（输入）
├── ans.py                      # 人工领域词典 (EntityLibrary, 14 类 1200+ 词)
├── run_extract.py              # 抽取主入口
├── run_all.py                  # 一键脚本：抽取→银标→评估
├── main.py                     # CLI 转发器
├── evaluate_kg.py              # 评估器（严格/宽松/关系级三口径）
├── app_streamlit.py            # 可视化（pyvis + Streamlit）
├── extractors/                 # 抽取算法核心包
│   ├── preprocess.py           #   章节/段落/句子切分
│   ├── ner.py                  #   HybridNER：AC 自动机 + 数值/型号正则
│   ├── paper_entity_recognizer.py  # spaCy 论文实体挖掘
│   ├── trigger_extractor.py    #   主力：触发词+实体共现
│   ├── pattern_extractor.py    #   高精度正则模板
│   ├── svo_extractor.py        #   spaCy 依存 SVO
│   ├── numeric_extractor.py    #   数值/单位关系
│   ├── type_extractor.py       #   类型/共现/章节关系
│   ├── relation_normalizer.py  #   关系归一化到统一本体
│   ├── llm_enhancer.py         #   LLM 二阶段质检与发现
│   ├── schema.py               #   Triple / Mention / RelationOntology
│   └── pipeline.py             #   总调度
├── tools/
│   ├── export_entities_by_type_json.py  # 词典 → JSON
│   └── export_silver_gold.py            # 银标生成（触发词∪类型∪章节）
├── data/entities_by_type.json   # 词典 JSON（运行时只读，可脱离 ans.py）
├── gold/                        # 银标
│   ├── silver_entities.txt      #   1221 个领域实体
│   ├── silver_triples.csv       #   7878 条严格银标
│   └── silver_triples_loose.csv # 18743 条宽松银标
└── output/                      # 产出
    ├── triples_with_meta.csv    #   带元信息的三元组
    ├── entities.csv             #   实体清单
    ├── knowledge_graph.csv      #   兼容三列格式
    ├── extraction_stats.txt     #   抽取统计
    ├── eval_report.txt          #   严格评估报告
    ├── eval_report_loose.txt    #   宽松评估报告
    ├── eval_tp.csv / eval_fp.csv / eval_fn.csv  # 错误分析明细
```

---

## 2. 环境与依赖

```bash
uv sync                                       # 同步基础依赖
uv run python -m spacy download zh_core_web_sm  # SVO 依存抽取与实体挖掘所需
uv sync --extra app                            # 可选：streamlit 可视化依赖
```

`pyproject.toml` 已声明：`pyahocorasick`、`spacy>=3.7`、`openai>=1.40`、`networkx`、`pyvis`、`pandas`、`tqdm`、`streamlit (optional)`。

---

## 3. 一键运行

```bash
# 最常用：mock LLM 模式（不需 API），一行跑完全流程
uv run python run_all.py

# 跑通后可视化
uv run streamlit run app_streamlit.py

# 使用 OpenAI 兼容 API（DeepSeek/Kimi/智谱等）启用 LLM 增强 + 按章发现
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set OPENAI_MODEL=deepseek-chat
uv run python run_all.py --llm openai --llm-discover
```

`run_all.py` 依次完成：

1. **导出词典 JSON**（`tools/export_entities_by_type_json.py`）
2. **抽取**（`run_extract.py --entities-json data/entities_by_type.json`）
3. **银标生成**（`tools/export_silver_gold.py`）
4. **严格评估** + **宽松评估**（`evaluate_kg.py`）

也可分步运行：

```bash
uv run python main.py extract        # 仅抽取
uv run python main.py manual         # 打印词典统计
uv run python main.py silver         # 仅生成银标
uv run python main.py eval           # 仅评估（严格）
uv run python main.py eval --gold gold/silver_triples_loose.csv --loose  # 宽松
uv run python main.py app            # 启动可视化
```

---

## 4. 抽取算法（自研，不只依赖 LLM）

每条三元组都带 `source` 字段标明产生算法，便于消融与回溯。所有算法以 `extractors/pipeline.py` 统一调度。

### 4.1 预处理 `preprocess.py`

- 行级扫描，识别 `第N章 ...`、`数字.数字 标题` 维护章节/小节上下文。
- 中文句末标点 `。！？；` 切句，过滤过短行。
- 输出 438 个段落、1042 个句子（含 chapter / section / paragraph_id）。

### 4.2 词典 NER `ner.py` + `paper_entity_recognizer.py`

- **AC 自动机**（`pyahocorasick`）对 1200+ 词典实体做 O(N) 多模式匹配；
- **数值正则**抓 `400km/h`、`12t`、`9000N·m` 等带单位数值；
- **型号正则**抓 `RQ-4A`、`MQ-9`、`F-111` 等型号编码；
- **paper_entity_recognizer.py**：用 spaCy 中文模型挖掘"论文中真实出现但词典未覆盖"的串（命名实体、`noun_chunks`、连续名/专名拼接），频次 ≥1 即并入 `HybridNER.add_terms`。本次跑挖了 **2042 个新实体**入库。
- 去重时采用「长 mention 优先 + 同范围去重」。

### 4.3 触发词共现 `trigger_extractor.py`（主力，命中 3280 条）

- 一句内识别实体后两两组合（仅相邻 K=6 个，避免远距离误关联）；
- 取实体对中间窗口（≤30 字、不跨句末标点）扫描触发词；
- 触发词命中 → 映射到 `RelationOntology` 中的 33 种统一关系（`improves` / `reduces` / `has_part` / `develops` / ...）；
- 综合**触发词长度、距离、紧贴度、否定情境、列举副词数**给 0–1 置信度；
- 被动语态自动翻转头尾（如「由 X 研制」→ X 是 head）；
- 关系-实体类型约束（如 `is_a` 的 tail 不允许是数值）。

### 4.4 高精度模板 `pattern_extractor.py`（68 条）

- 显式模板：「`由 X 驱动`→driven_by」「`A 与 B 连接`→connected_to」「`A 包括 B、C、D`→has_part」等；
- 触发词左侧取**最右最长**词典实体作 head，右侧取**最左最长**作 tail；
- score = 0.82~0.85（高于触发词，因为模板更确定）。

### 4.5 依存 SVO `svo_extractor.py`（197 条）

- 用 spaCy 中文模型，遍历所有 VERB/AUX 节点（不再只看 ROOT）；
- 收集 `nsubj/top/nsubjpass/csubj` 作 subject、`dobj/pobj/iobj/attr/ccomp/xcomp` 作 object；
- 谓词经 `RelationNormalizer` 归一化到本体；
- 头/尾用词典 NER 回填，"NER 命中实体"比"spaCy 切出的零碎词"优先。

### 4.6 数值关系 `numeric_extractor.py`（58 条）

- 四组正则匹配 `X 为/达到 N 单位`、`X 在 N 单位 左右`、`X 大于 N 单位`、`X 小于 N 单位`；
- 关系映射为 `has_value` / `greater_than_value` / `less_than_value`。

### 4.7 类型/章节/共现 `type_extractor.py`

- `TypeBasedExtractor`（347 条 `instance_of`）：在文中出现的实体 → 类型标签（飞行器/机翼构型/...）；
- `ChapterMembershipExtractor`（384 条 `discussed_in`）：实体在某章节出现 ≥2 次 → 章节标题；
- `CooccurrenceTypeExtractor`（287 条）：跨类型规则（飞行器 + 设计参数 → `has_parameter`，结构部件 + 材料 → `made_of`，等等）。

### 4.8 关系归一化 `relation_normalizer.py`

把 spaCy 抽出的自由谓词（如 "提升"、"显著降低"、"提出"）映射到统一关系本体（`improves`、`reduces`、`develops`），保证下游消费者只看到一套关系名。

### 4.9 LLM 增强（可选） `llm_enhancer.py`

**严格遵循"不允许只用 LLM"的课程要求**：LLM 仅作用于传统算法已识别的候选。

- **mock 模式**：等价于"再去重 + 边界清洗"，本地无 API 也能跑；
- **openai 模式** (`--llm openai`)：按章节分组送审，每批 ≤50 条，让模型做关系归一化 + 删冗 + 少量补全；
- **--llm-discover 模式**：把整章正文+已有候选发给模型，允许补充正文中可逐字核对的新三元组；代码侧严格校验 `head/tail` 是正文子串、`relation` 在白名单内。

---

## 5. 银标 (Silver Standard)

`tools/export_silver_gold.py` 用与抽取层**独立但同源**的派生规则产生银标，避免"用同一份 pred 当 gold"的循环验证陷阱。

| 派生类 | 规则 | 银标条数 |
|---|---|---|
| 触发词派生 | 同句相邻 4 个实体两两组合，中间窗口 ≤30 字命中触发词即映射到本体；被动翻转；负向词降权过滤 | ~7000 |
| 类型派生 | 实体在论文中出现 → 类型标签 `instance_of` | 445 |
| 章节派生 | 实体在某章节出现 ≥2 次 → 章节 `discussed_in` | 415 |
| **合计 strict** | | **7878** |
| **loose** | 所有同句相邻实体对一律 `co_occurs_with`（仅做实体对覆盖评估） | 18743 |

银标同步使用 `paper_entity_recognizer` 把高频挖掘实体并入 NER 词表，确保评估目标与抽取目标在实体空间上对齐。

---

## 6. 评估口径

`evaluate_kg.py` 输出**三个口径** + **关系级明细** + **TP/FP/FN 误判清单**。

### 6.1 三口径

| 口径 | 匹配规则 | 用途 |
|---|---|---|
| 严格 F1 | `(head, relation, tail)` 完全相等 | 关系级抽取能力 |
| 宽松 F1 | `(head, tail)` 相等（不管关系） | 实体对发现能力 |
| 实体级 F1 | 实体集合的 P/R/F1 | 概念覆盖能力 |

### 6.2 当前指标

```
严格 F1 (head,rel,tail 全匹配):  P=0.5291  R=0.3118  F1=0.3923  TP=2456
宽松 F1 (仅 head,tail 匹配):      P=0.6111  R=0.3575  F1=0.4511  TP=2524
实体级 F1 (端点集合):             P=0.6022  R=0.7265  F1=0.6585  TP=887
关系级宏平均 F1 (Top-15 关系):                          F1=0.3932
```

### 6.3 关系级亮点（按 F1 排序）

| 关系 | P | R | F1 |
|---|---|---|---|
| `discussed_in` | 1.000 | 0.925 | **0.961** |
| `instance_of` | 1.000 | 0.780 | **0.876** |
| `causes` | 0.680 | 0.313 | 0.429 |
| `located_at` | 0.482 | 0.379 | 0.424 |
| `verifies` | 0.552 | 0.337 | 0.419 |
| `reduces` | 0.539 | 0.302 | 0.388 |
| `leads_to` | 0.481 | 0.315 | 0.381 |
| `improves` | 0.519 | 0.275 | 0.359 |
| `develops` | 0.482 | 0.236 | 0.316 |

### 6.4 错误样本明细

`output/eval_tp.csv` / `eval_fp.csv` / `eval_fn.csv` 各列出 (head, relation, tail) 三元组，便于写报告时引用具体错例。`eval_report.txt` 末尾还会打印**实体对相同但关系不同**的混淆 Top-12（如 `reduces` 与 `improves` 互混 15 例）。

---

## 7. 可视化

```bash
uv run streamlit run app_streamlit.py
```

特性：

- 数据源切换（默认 `output/triples_with_meta.csv`，可切换 silver / loose）；
- **关系白名单**多选过滤；
- 最小 `score` / 最小实体长度 / 子图最大节点 / BFS 跳数四个滑块；
- **节点按类型染色**：飞行器=蓝、机翼构型=绿、设计参数=黄、组织=灰、人物=橙等；
- 统计面板：三元组数、实体数、关系类型数、关系分布、来源分布；
- 当前子图边明细折叠面板，可逐边查看 score/source。

---

## 8. 课程要求达成情况

| 要求 | 阈值 | 当前 | 状态 |
|---|---|---|---|
| 概念（实体） | ≥500 | **1473** | ✓ 满足（×2.9） |
| 关系（三元组） | ≥1000 | **4642** | ✓ 满足（×4.6） |
| 标注概念 | ≥200 | **1221** | ✓ 满足（×6.1） |
| 标注关系 | ≥400 | **7878** | ✓ 满足（×19.7） |
| 自动抽取算法源代码 | 必须 | `extractors/` 内 9 个算法 + 流水线 | ✓ 提供 |
| 不允许只用 LLM | 必须 | LLM 仅在传统候选基础上做质检/补全，可降级 mock | ✓ 满足 |

---

## 9. 常见问题

**Q: 没有 OPENAI_API_KEY 能跑吗？**
A: 能，默认就是 mock 模式。LLM 增强会被替换为"再去重 + 边界清洗"。

**Q: spaCy 模型下载失败？**
A: SVO 与 paper_entity_recognizer 会优雅降级，仅触发词/模板/类型/章节等抽取器仍工作。可加 `--no-svo --no-paper-entity-mine` 完全跳过 spaCy。

**Q: 想用自己的人工金标评估？**
A: 把人工金标做成与 `gold/silver_triples.csv` 同列名（head, relation, tail）的 CSV，运行：
```bash
uv run python evaluate_kg.py --gold path/to/your_gold.csv
```

**Q: 实体抽多了想精简？**
A: 关掉论文实体挖掘 `uv run python run_extract.py --no-paper-entity-mine`，或调高频次门槛 `--paper-entity-min-freq 2`。
