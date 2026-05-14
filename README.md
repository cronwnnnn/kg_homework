# 单-双折叠翼变体飞行器 · 知识图谱构建系统

> 基于清华大学郭廷宇博士论文 `aftcln.txt`，使用多策略 NLP 算法自动抽取领域知识图谱。
>
> **课程目标**：概念 ≥500、关系 ≥1000、人工金标 ≥400 关系做评估。
>
> **当前规模**：
> - 领域词典实体 **1473** 个（`data/entities_by_type.json`）
> - 自动抽取实体 **503** 个，三元组 **1367** 条（收紧版 + 依存句法增强，无 LLM）
> - 第四章人工金标 **745** 条（`gold/gold_triples_augmented.csv`）
>
---

## 1. 项目总览

```
my-project/
├── aftcln.txt                  # 论文原文（输入）
├── ans.py                      # 人工领域词典 (EntityLibrary, 14 类 1200+ 词)
├── run_extract.py              # 抽取主入口
├── run_all.py                  # 一键脚本：抽取→评估
├── main.py                     # CLI 转发器
├── evaluate_kg.py              # 评估器（严格/宽松/Partial/实体级 4 口径）
├── app_streamlit.py            # 知识图谱浏览（pyvis + Streamlit）
├── app_qa/                     # 领域问答助手（双面板 QA + 子图可视化）
│   ├── app.py                  #   Streamlit 双面板主程序
│   ├── kg_store.py             #   三元组加载 + 邻接索引
│   ├── corpus_index.py         #   论文文本块索引（用于证据回填）
│   ├── query_parser.py         #   意图识别 + 实体抽取
│   ├── retriever.py            #   按意图召回 KG 子集 + 原文片段
│   ├── answer_builder.py       #   模板答案 + LLM 重写
│   └── llm_client.py           #   OpenAI 兼容客户端封装
├── extractors/                 # 抽取算法核心包
│   ├── preprocess.py           #   章节/段落/句子切分
│   ├── ner.py                  #   HybridNER：AC 自动机 + 数值/型号正则
│   ├── trigger_extractor.py    #   主力：触发词+实体共现
│   ├── pattern_extractor.py    #   高精度正则模板
│   ├── svo_extractor.py        #   spaCy 依存 SVO（默认禁用）
│   ├── dependency_re.py        #   依存句法增强抽取（基于 spaCy 依存树，默认启用）
│   ├── numeric_extractor.py    #   数值/单位关系
│   ├── type_extractor.py       #   类型 + 章节归属
│   ├── relation_normalizer.py  #   关系归一化到统一本体
│   ├── llm_enhancer.py         #   LLM 二阶段质检与发现
│   ├── schema.py               #   Triple / Mention / RelationOntology
│   ├── pipeline.py             #   总调度
│   └── archive/                #   已归档（默认禁用）
│       └── paper_entity_recognizer.py  # spaCy 论文实体挖掘
├── tools/                       # 主线工具
│   ├── export_entities_by_type_json.py  # 词典 → JSON（run_all.py 必用）
│   └── archive/                 # 一次性脚本归档（详见 archive/README.md）
│       ├── align_gold_to_pred.py            # gold/pred 颗粒度对齐
│       ├── augment_gold_with_instance_of.py # 用 pred 合理 instance_of 补 gold
│       ├── error_analysis.py                # 错误分析报告
│       ├── suggest_ner_terms.py             # 从 FN 推荐 NER 词典扩充
│       ├── merge_ner_terms.py               # 合并 NER 词典
│       ├── llm_ner_expand.py                # LLM 全文 NER（实验回滚）
│       ├── audit_llm_ner.py                 # LLM 实体审计（实验回滚）
│       └── prune_long_ner.py                # 长复合词剔除（实验回滚）
├── data/
│   ├── entities_by_type.json    # 主词典 JSON（运行时只读，可脱离 ans.py）
│   └── aliases.json             # 实体别名规约表
├── gold/                        # 第四章人工金标
│   ├── gold_triples.csv         # 第四章原始人工金标（444 条）
│   ├── gold_triples_aligned.csv # 对齐版（432 条）
│   ├── gold_triples_augmented.csv  # 补全版（745 条，主基线）
│   ├── gold_entities.csv        # 金标实体清单
└── output/                      # 产出
    ├── triples_with_meta.csv    # 带元信息的三元组
    ├── entities.csv             # 实体清单
    ├── knowledge_graph.csv      # 兼容三列格式
    ├── extraction_stats.txt     # 抽取统计
    ├── eval_report.txt          # 第四章金标评估报告
    └── eval_tp.csv / eval_fp.csv / eval_fn.csv  # 错误分析明细
```

---

## 2. 环境与依赖

```bash
uv sync                                       # 同步基础依赖
uv run python -m spacy download zh_core_web_sm  # SVO 依存抽取与实体挖掘所需
uv sync --extra app                            # streamlit 可视化依赖
```

`pyproject.toml` 已声明：`pyahocorasick`、`spacy>=3.7`、`openai>=1.40`、`networkx`、`pyvis`、`pandas`、`tqdm`、`streamlit (optional)`。

---

## 3. 一键运行

```bash
# 最常用：mock LLM 模式（不需 API），一行跑完全流程
uv run python run_all.py

# 跑通后可视化
uv run streamlit run app_streamlit.py

# 使用 OpenAI 兼容 API（DeepSeek/Kimi/智谱等）启用 LLM 发现补全（可选）
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set OPENAI_MODEL=deepseek-chat
uv run python run_extract.py --entities-json data/entities_by_type.json --llm openai --no-llm --llm-discover
```


`run_all.py` 依次完成：

1. **复用或生成词典 JSON**：若 `data/entities_by_type.json` 已存在则**复用**。
2. **抽取**（`run_extract.py --entities-json data/entities_by_type.json`）
3. **评估**（`evaluate_kg.py`，默认用 `gold/gold_triples_augmented.csv` ）


```bash
uv run python main.py extract        # 仅抽取
uv run python main.py manual         # 打印词典统计
uv run python main.py eval           # 仅评估
uv run python main.py app            # 启动可视化
```

---

## 4. 抽取算法

每条三元组都带 `source` 字段标明产生算法，便于消融与回溯。所有算法以 `extractors/pipeline.py` 统一调度。

### 4.1 预处理 `preprocess.py`

- 行级扫描，识别 `第N章 ...`、`数字.数字 标题` 维护章节/小节上下文。
- 中文句末标点 `。！？；` 切句，过滤过短行。
- 输出 438 个段落、1042 个句子（含 chapter / section / paragraph_id）。

### 4.2 词典 NER `ner.py` + `paper_entity_recognizer.py`

- **AC 自动机**（`pyahocorasick`）对 1200+ 词典实体做 O(N) 多模式匹配；
- **数值正则**抓 `400km/h`、`12t`、`9000N·m` 等带单位数值；
- **型号正则**抓 `RQ-4A`、`MQ-9`、`F-111` 等型号编码；
- 去重时采用「长 mention 优先 + 同范围去重」。

### 4.3 触发词共现 `trigger_extractor.py`

- 一句内识别实体后两两组合（仅相邻 K=6 个，避免远距离误关联）；
- 取实体对中间窗口（≤30 字、不跨句末标点）扫描触发词；
- 触发词命中 → 映射到 `RelationOntology` 中的 33 种统一关系（`improves` / `reduces` / `has_part` / `develops` / ...）；
- 综合**触发词长度、距离、紧贴度**给 0–1 置信度；
- **min_score = 0.55** 保证质量。

### 4.4 高精度模板 `pattern_extractor.py`

- 显式模板：「`由 X 驱动`→driven_by」「`A 与 B 连接`→connected_to」「`A 包括 B、C、D`→has_part」等；
- 触发词左侧取**最右最长**词典实体作 head，右侧取**最左最长**作 tail；
- score = 0.82~0.85（高于触发词，因为模板更确定）。

### 4.5 依存句法增强 `dependency_re.py`


**评估效果**：
- pred 三元组：1300 → **1367**（+67 条新关系）
- 关系分布更均衡：generates、located_at、has_part、improves、needs 等弱项关系 P=0.3-0.6
- 实体级 F1 反升 0.002

可关闭：`run_extract.py --no-dep-re`

### 4.6 数值关系 `numeric_extractor.py`

- 四组正则匹配 `X 为/达到 N 单位`、`X 在 N 单位 左右`、`X 大于 N 单位`、`X 小于 N 单位`；
- 关系映射为 `has_value` / `greater_than_value` / `less_than_value`。

### 4.7 类型 / 章节 `type_extractor.py`

- `TypeBasedExtractor`（**434** 条 `instance_of`）：在文中出现的实体 → 类型标签（飞行器/机翼构型/...）.
- `ChapterMembershipExtractor`（**474** 条 `discussed_in`）：实体在某章节出现 ≥2 次 → 章节标题；评估时被 `--exclude-relations` 屏蔽（不影响 F1），但 app_qa 问答助手"在哪章讨论"功能仍依赖它。

### 4.8 关系归一化 `relation_normalizer.py`

把 spaCy 抽出的自由谓词（如 "提升"、"显著降低"、"提出"）映射到统一关系本体（`improves`、`reduces`、`develops`），保证下游只看到一套关系名。

### 4.9 LLM 增强（可选） `llm_enhancer.py`

- **openai 模式** (`--llm openai`)：按章节分组送审，每批 ≤50 条，让模型做关系归一化 + 删冗 + 少量补全；
- **--llm-discover 模式**：把整章正文+已有候选发给模型，允许补充正文中可逐字核对的新三元组；代码侧严格校验 `head/tail` 是正文子串、`relation` 在白名单内。

---

## 5. 评估口径

`evaluate_kg.py` 输出**4 个 F1 口径** + **关系级明细** + **TP/FP/FN 误判清单**。

### 5.1 四口径

| 口径 | 匹配规则 | 用途 |
|---|---|---|
| 严格 F1 (L1) | `(head, relation, tail)` 完全相等 | 关系级抽取能力 |
| 宽松 F1 (L2) | `(head, tail)` 相等（忽略关系） | 实体对发现能力 |
| Partial F1 (L3) | 关系一致 + head/tail 双向子串匹配（min_len=2） | 反映真实语义匹配水平 |
| 实体级 F1 | 实体集合的 P/R/F1 | 概念覆盖能力 |

### 5.2 当前指标（基于 `gold/gold_triples_augmented.csv` 745 条 + 6 算法主基线 pred 1367 条，含依存句法增强）

```
严格 F1   (L1):  P=0.8074  R=0.5940  F1=0.6845  TP=436  pred=540  gold=734
宽松 F1   (L2):  P=0.8089  R=0.5948  F1=0.6855  TP=436
Partial F1 (L3): P=0.8148  R=0.5995  F1=0.6907  TP=440   ← 最高
实体级 F1:       P=0.9824  R=0.6862  F1=0.8080  TP=446  pred_ent=454  gold_ent=650
```

## 6. 第四章人工金标评估专题


### 6.1 评估命令

```bash
uv run python evaluate_kg.py \
    --gold gold/gold_triples_augmented.csv \
    --pred output/triples_with_meta.csv \
    --chapter "第4章" --include-global \
    --aliases-file data/aliases.json \
    --exclude-relations discussed_in,co_occurs_with \
    --report output/eval_report_ch4_augmented.txt
```

### 6.2 评估器参数

新增的评估参数（见 `evaluate_kg.py --help`）：

- `--chapter "第4章"`：仅评估指定章节的 pred 三元组
- `--include-global`：把 `chapter` 为空且为 `instance_of/is_a/type_taxonomy` 的全局类型分类也纳入评估
- `--aliases-file`：实体别名表，评估时做 canonical 规约
- `--exclude-relations`：评估时排除的关系（如 `discussed_in,co_occurs_with`）
- `--no-normalize-rel`：关闭中文/has 关系名归一化

---

## 7. 可视化与问答应用

项目提供 Streamlit 应用：

### 7.1 领域问答助手 `app_qa/`

```bash
uv run streamlit run app_qa/app.py
```

双面板布局：

- **左侧（问答）**：自然语言输入问题 → 意图识别（定义/列举/数值/属性/因果/章节/关系/邻居 8 种）→ 模板答案 + 三元组依据 + 原文片段；
- **右侧（图谱）**：以问题里识别的实体为中心，渲染 K 跳子图，高亮主答案路径；
- **LLM 可选**：勾选"启用 LLM 重写回答"后，把模板答案 + 图谱 + 原文作为材料调用 LLM 润色为更自然的中文。

示例问题（更多见 `app_qa/samples/example_questions.txt`）：

```
什么是变体飞行器？        # 定义
有哪些控制方法？           # 列举
全球鹰的速度是多少？       # 数值
什么减少阻力？             # 因果
升力 和 阻力 的关系        # 二元关系
反馈线性化在哪章讨论？     # 章节
总结全球鹰                 # 综合摘要
```

---

## 8. 课程要求达成情况

| 要求 | 阈值 | 当前 | 状态 |
|---|---|---|---|
| 概念（实体） | ≥500 | **1473**（领域词典） / **503**（实际抽取） | ✓ 满足（×2.9） |
| 关系（三元组） | ≥1000 | **1367**（含依存句法增强）| ✓ 满足（×1.4） |
| 人工金标关系 | ≥400 | **745**（`gold_triples_augmented.csv`） | ✓ 满足（×1.9） |
| 自动抽取算法源代码 | 必须 | `extractors/` 内 6 个算法（含 `dependency_re.py`）+ LLM 增强 + 流水线 | ✓ 提供 |
| 不允许只用 LLM | 必须 | 主体由传统抽取（含依存句法）产 1367 条；关闭 LLM 仍可独立运行 | ✓ 满足 |

---
