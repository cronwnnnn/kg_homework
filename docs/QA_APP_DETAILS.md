# 智能问答应用 `app_qa/` 详解

> 配套实验报告使用。本文档详细讲解项目中"领域问答助手"（app_qa）的设计思路、模块划分、检索与答案生成流程、可视化方案，以及与知识图谱、原文语料、LLM 的协同方式。

> **关联文档**：
> - [ALGORITHM_DETAILS.md](ALGORITHM_DETAILS.md)：6 个抽取算法（生成本应用所用的三元组）
> - [NER_DETAILS.md](NER_DETAILS.md)：实体识别（query_parser 复用同一份 HybridNER）

---

## 目录

- [1. 应用定位与整体架构](#1-应用定位与整体架构)
- [2. 7 个模块的分工](#2-7-个模块的分工)
- [3. 模块详解](#3-模块详解)
- [4. 端到端流程举例](#4-端到端流程举例)
- [5. 知识图谱可视化方案](#5-知识图谱可视化方案)
- [6. LLM 协同与降级策略](#6-llm-协同与降级策略)
- [7. 缓存与性能优化](#7-缓存与性能优化)
- [8. 启动与使用](#8-启动与使用)
- [9. 局限与未来优化](#9-局限与未来优化)

---

## 1. 应用定位与整体架构

**一句话定位**：基于知识图谱 + 论文原文的"双面板问答助手"。

- 左侧：自然语言 QA（解析问题 → 模板答案 + 三元组依据 + 原文证据）
- 右侧：知识图谱可视化（以问题中实体为中心展开 K 跳子图，高亮主答案路径）

### 1.1 双面板布局

```
┌─────────────────────────────────────────────────────────────────────┐
│ ✈ 单-双折叠翼变体飞行器 · 领域问答助手                              │
│ 基于知识图谱（NER + 触发词 + 模板 + 依存增强 + 类型/章节）与论文原文 │
├──────────────┬──────────────────────────────────────────────────────┤
│  Sidebar     │  ▲ 顶部统计指标条（实体数/三元组数/关系类型数）       │
│              │                                                       │
│  数据源      │  ── 示例问题（折叠面板，点击复用）──                   │
│  ─────────   │                                                       │
│  三元组CSV   │  ┌─ 问题输入框 ────────────────────────────────┐      │
│  论文文本    │  │  请输入您的问题...                          │      │
│              │  └─────────────────────────────────────────────┘      │
│  可视化参数  │                                                       │
│  ─────────   │  ┌────── 左面板 ──────┐ ┌────── 右面板 ──────┐         │
│  BFS跳数 ●━○ │  │ 解析结果           │ │ 知识图谱可视化     │         │
│  节点数 ●━━○ │  │  意图: definition  │ │                    │         │
│              │  │  识别实体: 全球鹰   │ │  ●━━━━●  pyvis    │         │
│  LLM增强     │  │                    │ │  │     │  网络图  │         │
│  ─────────   │  │ 回答（模板/LLM）   │ │  ●━━━━●            │         │
│  ☐ 启用LLM   │  │ 主答案三元组       │ │   ●━●              │         │
│              │  │  • A —rel→ B      │ │                    │         │
│  LLM 配置    │  │                    │ │ 中心: 全球鹰        │         │
│  ─────────   │  │ 更多相关三元组(折) │ │ 跳数: 2 | 节点: 18 │         │
│  API Key     │  │                    │ │                    │         │
│  Base URL    │  │ 原文证据           │ │ 图例·颜色/类型(折)  │         │
│  Model       │  │  [4.2节] xxx...    │ │                    │         │
│              │  │  [4.3节] yyy...    │ │                    │         │
└──────────────┴────────────────────────┴────────────────────────────┘
```

### 1.2 核心管道（数据流）

```
                 ┌──────── 用户问题 ────────┐
                 ▼                            │
        ┌───────────────────┐                 │
        │  query_parser.py  │ 正则模板 + NER  │
        │  → ParsedQuery    │ (12种意图 + 实体)│
        └─────────┬─────────┘                 │
                  ▼                            │
        ┌───────────────────┐ ┌─────────────┐ │
        │   retriever.py    │─│ kg_store.py │ │
        │ 按意图选关系/路径  │ │ networkx图 +│ │
        │ → RetrievalResult │ │ 倒排索引     │ │
        └─────────┬─────────┘ └─────────────┘ │
                  │            ┌─────────────┐ │
                  └───────────▶│ corpus_index│ │
                               │ 原文句子检索 │ │
                               └─────────────┘ │
                  ▼                            │
        ┌───────────────────┐ ┌─────────────┐ │
        │ answer_builder.py │─│ llm_client  │ │
        │ 模板合成 + LLM重写 │ │ OpenAI兼容  │ │
        └─────────┬─────────┘ └─────────────┘ │
                  ▼                            │
        ┌──── app.py (Streamlit 主程序) ───────┐
        │ 左面板：解析/回答/三元组/原文证据    │
        │ 右面板：pyvis 子图（中心+主答案高亮） │
        └──────────────────────────────────────┘
```

---

## 2. 7 个模块的分工

| 文件 | 大小 | 角色 | 关键接口 |
|---|---|---|---|
| `app_qa/app.py` | 16 KB | Streamlit 主程序 + 可视化渲染 | `main()` / `to_pyvis_html()` |
| `app_qa/kg_store.py` | 7.7 KB | 知识图谱内存视图 + 倒排索引 | `KGStore.neighbors_subgraph()` |
| `app_qa/corpus_index.py` | 3.4 KB | 论文原文句子索引 | `Corpus.search()` / `search_pair()` |
| `app_qa/query_parser.py` | 6.3 KB | 意图识别 + 实体抽取 | `QueryParser.parse()` |
| `app_qa/retriever.py` | 5.9 KB | 按意图召回 KG + 原文 | `retrieve(parsed, kg, corpus)` |
| `app_qa/answer_builder.py` | 5.8 KB | 模板合成 + 可选 LLM 重写 | `build_answer(parsed, retr, llm)` |
| `app_qa/llm_client.py` | 3.6 KB | OpenAI 兼容客户端（含新旧 API 兼容） | `LLMClient.complete(prompt)` |

> 设计准则：每个模块单一职责，模块间通过 dataclass 数据结构通信，便于单元测试和替换实现。

---

## 3. 模块详解

### 3.1 `query_parser.py` — 意图识别 + NER

**核心问题**：把自然语言问题转换成结构化的 `ParsedQuery`。

#### 12 种意图（`QueryIntent` 枚举）

| 意图 | 含义 | 示例问题 |
|---|---|---|
| `DEFINITION` | X 是什么 / 什么是 X | "什么是变体飞行器？" |
| `LIST_INSTANCE` | 有哪些 X / 列举 X | "有哪些控制方法？" |
| `VALUE` | X 是多少 / X 的值 | "全球鹰的速度是多少？" |
| `PROPERTY` | X 的 Y / X 有哪些 Y | "变体飞行器有哪些参数？" |
| `CAUSE` | 什么导致 X / 什么减少 X | "什么减少阻力？" |
| `EFFECT` | X 导致什么 / X 影响什么 | "折叠翼会影响什么？" |
| `LOCATION` | X 在哪 / X 位于哪 | "气动布局位于哪？" |
| `CHAPTER` | X 在哪章讨论 | "反馈线性化在哪章讨论？" |
| `RELATION` | X 和 Y 的关系 | "升力 和 阻力 的关系" |
| `NEIGHBOR` | 与 X 相关的 | "与机翼相关的有哪些" |
| `SUMMARY` | 总结 X / X 概述 | "总结全球鹰" |
| `UNKNOWN` | 兜底（按 NEIGHBOR 处理） | 默认 |

#### 解析算法（3 步）

1. **正则模板匹配**（23 条规则，每条带 `target_relation` 提示）：
   ```python
   (QueryIntent.DEFINITION, re.compile(r"^(.+?)是什么(\?|？)?$"), "instance_of"),
   (QueryIntent.VALUE,      re.compile(r"(.+?)(?:是)?多少(\?|？)?$"), "has_value"),
   (QueryIntent.CAUSE,      re.compile(r"(?:什么|哪些).*?(?:导致|引起|造成)(.+?)$"), "causes"),
   (QueryIntent.RELATION,   re.compile(r"(.+?)(?:和|与|跟)(.+?)(?:的关系|的联系)"), None),
   # ... 共 23 条
   ```

2. **复用 `HybridNER` 做实体抽取**（数值正则关闭，型号正则保留）：
   ```python
   self.ner = HybridNER(vocab=vocab, enable_numeric=False, enable_aircraft_code=True)
   ```

3. **模板捕获组实体置顶**：如果模板组捕获到的字符串里有词典实体，把它提到 `entities` 列表第一位（确保 `primary` 是正确的中心实体）。

#### 输出

```python
@dataclass
class ParsedQuery:
    raw: str                          # 清洗后的问题
    intent: QueryIntent               # 意图
    entities: list[str]               # 实体列表
    target_relation: str | None       # 目标关系提示

    @property
    def primary(self) -> str:         # 第一个实体（子图中心）
        return self.entities[0] if self.entities else ""

    @property
    def secondary(self) -> str:       # 第二个实体（用于 RELATION）
        return self.entities[1] if len(self.entities) >= 2 else ""
```

---

### 3.2 `kg_store.py` — 知识图谱内存视图

**核心问题**：把三元组 CSV 加载为图谱 + 倒排索引，提供高效的图查询接口。

#### 数据结构

```python
@dataclass(frozen=True)
class TripleRow:
    head: str
    relation: str
    tail: str
    score: float = 1.0
    source: str = ""        # type / trigger / pattern / dep / ...
    chapter: str = ""
    sentence: str = ""
    trigger: str = ""

class KGStore:
    self.graph: nx.MultiDiGraph                # 主图
    self._by_head:     dict[str, list[Triple]] # 按 head 倒排
    self._by_tail:     dict[str, list[Triple]] # 按 tail 倒排
    self._by_relation: dict[str, list[Triple]] # 按关系倒排
    self._by_pair:     dict[(str,str), list]   # 按实体对倒排
```

**4 个倒排索引**让所有常见查询都是 O(1) 哈希。

#### 关键方法

| 方法 | 用途 | 复杂度 |
|---|---|---|
| `triples_of(entity)` | 实体作为头或尾的所有三元组 | O(出现次数) |
| `out_edges(entity, relation)` | 该实体出边（可过滤关系） | O(出度) |
| `in_edges(entity, relation)` | 该实体入边 | O(入度) |
| `triples_with_relation(rel)` | 某关系的所有三元组 | O(关系频次) |
| `related(a, b)` | 两实体间所有直接关系 | O(1) |
| `find_paths(a, b, max_hops)` | 多跳路径搜索（无向） | O(图大小) |
| `neighbors_subgraph(entity, hops, max_nodes)` | BFS K 跳子图 | O(节点数) |

#### CSV 表头兼容

```python
head = (r.get("head") or r.get("头实体") or "").strip()
rel  = (r.get("relation") or r.get("关系") or "").strip()
tail = (r.get("tail") or r.get("尾实体") or "").strip()
```

无论 `output/triples_with_meta.csv`（英文表头）还是 `gold/gold_triples_augmented.csv`（中文表头）都能直接加载。

#### BFS 子图（核心可视化算法）

```python
def neighbors_subgraph(self, entity, hops=2, max_nodes=60):
    nodes = {entity}
    frontier = {entity}
    for _ in range(hops):
        if len(nodes) >= max_nodes: break
        nxt = set()
        for u in frontier:
            for v in self.graph.successors(u):     # 顺着边
                if v not in nodes: nxt.add(v)
            for v in self.graph.predecessors(u):   # 逆着边
                if v not in nodes: nxt.add(v)
        frontier = nxt
        for x in frontier:
            nodes.add(x)
            if len(nodes) >= max_nodes: break
    return self.graph.subgraph(nodes).copy()
```

**关键**：双向 BFS（不管节点是头还是尾，邻居都拉进来）+ max_nodes 截断（防止"飞行器"这种明星节点把全图都拽进来）。

---

### 3.3 `corpus_index.py` — 论文原文句子索引

**核心问题**：在原文中检索包含实体的句子，作为答案证据。

#### 数据结构

```python
@dataclass(frozen=True)
class SentenceRecord:
    sid: int                # 句子 ID
    text: str               # 句子文本
    chapter: str            # 所属章节（如"第4章"）
    section: str            # 所属小节（如"4.2.1"）
    paragraph_id: int
```

#### 关键接口

```python
class Corpus:
    def search(self, entity, limit=20):           # 包含该实体的句子
    def search_any(self, *entities, limit=20):    # 包含任一实体的句子
    def search_all(self, *entities, limit=20):    # 同时包含所有实体的句子
    def search_pair(self, a, b, limit=10):        # 同时包含 a 和 b，按短句优先
```

#### 懒加载倒排索引

```python
self._cached_index: dict[str, list[int]] = {}    # 按需缓存

def search(self, entity, limit=20):
    if entity in self._cached_index:
        idxs = self._cached_index[entity]
    else:
        idxs = [i for i, s in enumerate(self.sentences) if entity in s.text]
        self._cached_index[entity] = idxs
    return [self.sentences[i] for i in idxs[:limit]]
```

> 设计权衡：预构建全词典倒排会占内存（1473 词 × 1042 句 × 平均 hit 数）；懒加载按查询次数线性增长，对 QA 场景刚好（同一会话不会查太多不同实体）。

---

### 3.4 `retriever.py` — 意图驱动检索

**核心问题**：根据 `ParsedQuery` 的 intent，从 KG 和 Corpus 中选出最合适的三元组和证据。

#### 输出

```python
@dataclass
class RetrievalResult:
    intent: QueryIntent
    triples: list[TripleRow]              # 主答案候选
    evidence_rows: list[SentenceRecord]   # 原文证据
    subgraph_seed: str                    # 子图中心实体
    related_triples: list[TripleRow]      # 辅助显示（折叠面板）
    note: str = ""                        # 提示信息
```

#### 12 种意图的检索策略

```python
DEFINITION:
    rows = kg.out_edges(primary, "instance_of") + kg.out_edges(primary, "is_a")
    if not rows:
        for rel in ("has_part", "used_for", "develops", "applied_to"):
            rows.extend(kg.out_edges(primary, rel))
            if rows: break
    res.triples = sorted(rows, key=lambda t: -t.score)[:8]

LIST_INSTANCE:
    rows = [t for t in kg.triples_with_relation("instance_of") if t.tail == primary]
    if not rows:
        rows = [t for t in kg.triples_with_relation("instance_of") if primary in t.tail]
    res.triples = sorted(rows, key=lambda t: -t.score)[:15]

VALUE:
    for rel in ("has_value", "greater_than_value", "less_than_value", "equals_to"):
        rows.extend(kg.out_edges(primary, rel))
    res.triples = sorted(_dedup(rows), key=lambda t: -t.score)[:8]

CAUSE:   # 反向（哪些实体 → 影响 primary）
    rel_filter = (parsed.target_relation,) if parsed.target_relation else _CAUSE_RELATIONS
    for rel in rel_filter:
        rows.extend(kg.in_edges(primary, rel))

EFFECT:  # 正向（primary → 影响哪些实体）
    rel_filter = (parsed.target_relation,) if parsed.target_relation else _EFFECT_RELATIONS
    for rel in rel_filter:
        rows.extend(kg.out_edges(primary, rel))

LOCATION:    rows = kg.out_edges(primary, "located_at")
CHAPTER:     rows = kg.out_edges(primary, "discussed_in")

PROPERTY:
    if target_rel: rows = kg.out_edges(primary, target_rel)
    else:
        for rel in _PROPERTY_RELATIONS:   # has_part / has_value / has_parameter ...
            rows.extend(kg.out_edges(primary, rel))

RELATION + secondary:
    direct = kg.related(primary, secondary)         # 直接关系
    if not direct:
        paths = kg.find_paths(primary, secondary, max_hops=3, max_paths=4)
        # 展开多跳路径

SUMMARY / NEIGHBOR / UNKNOWN:
    rows = kg.triples_of(primary)
```

#### 原文证据召回

```python
if intent == RELATION and secondary:
    res.evidence_rows = corpus.search_pair(primary, secondary, limit=6)
    if not res.evidence_rows:
        res.evidence_rows = corpus.search(primary, limit=3) + corpus.search(secondary, limit=3)
else:
    res.evidence_rows = corpus.search(primary, limit=6)
```

#### 辅助"更多相关三元组"

```python
if res.subgraph_seed:
    all_of_primary = kg.triples_of(res.subgraph_seed)
    main_keys = {t.key for t in res.triples}
    res.related_triples = [t for t in all_of_primary if t.key not in main_keys][:10]
```

---

### 3.5 `answer_builder.py` — 模板合成 + LLM 重写

**核心问题**：把 `RetrievalResult` 渲染成自然语言答案，支持两种模式。

#### 模板路径

```python
def _template_compose(parsed, retr):
    lead = _intent_lead(parsed.intent, parsed.primary, parsed.secondary)
    bullets = [_fmt_triple(t) for t in retr.triples[:12]]
    body = [lead] + [f"  • {b}" for b in bullets]
    if retr.evidence_rows:
        body.append("原文佐证：")
        for ev in retr.evidence_rows[:3]:
            body.append(f"  · [{ev.section or ev.chapter}] {ev.text}")
    return Answer(text="\n".join(body), bullets=bullets, ...)
```

12 个意图各有一句**意图引导句**（`_intent_lead`）：

| 意图 | 引导句 |
|---|---|
| `DEFINITION` | "关于「X」是什么，知识图谱中给出的回答如下：" |
| `VALUE` | "「X」的相关数值/取值信息：" |
| `CAUSE` | "导致 / 影响「X」的因素：" |
| `RELATION` | "「X」与「Y」之间的关系：" |
| `SUMMARY` | "图谱中关于「X」的关键信息汇总：" |
| ... | ... |

#### LLM 路径（可选）

```python
def build_answer(parsed, retr, *, llm=None, use_llm=False):
    ans = _template_compose(parsed, retr)
    if not use_llm or llm is None or not llm.available: return ans
    if not retr.triples and not retr.evidence_rows:    return ans

    prompt = _build_llm_prompt(parsed, retr)
    enhanced = llm.complete(prompt)
    if enhanced.strip():
        ans.text = enhanced.strip()
        ans.llm_used = True
    return ans
```

**Prompt 模板**（节选）：

```
你是航空知识问答助手。请根据给定的「知识图谱三元组」与「论文原文证据」，对用户问题给出准确、简洁的中文回答。

要求：
1) 回答先用一两句话直接给出结论，再分点列出依据。
2) 仅使用「材料」中出现的信息；如材料不足，请明确说"图谱与论文未提供更多信息"。
3) 引用三元组时使用 (头实体 — 关系 → 尾实体) 的格式。
4) 不要编造，不要发挥与材料无关的内容。

用户问题：{question}
意图：{intent}
主要实体：{primary}
次要实体：{secondary}

知识图谱三元组：
{triples}

论文原文证据：
{evidence}

请给出回答：
```

**关键约束**：强制 LLM 只能引用材料中的信息，无中生有的回答会被材料约束克制。

---

### 3.6 `llm_client.py` — OpenAI 兼容客户端

**核心问题**：封装 OpenAI 协议调用，未配置时静默降级。

#### 兼容能力

```python
class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=60):
        self.api_key  = api_key  or os.environ.get("OPENAI_API_KEY")  or ""
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or ""
        self.model    = model    or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    @property
    def available(self) -> bool:
        return bool(self.api_key)
```

`available=False` 时 UI 自动隐藏 LLM 功能，避免空调用。

#### 新旧 API 兼容

```python
# 先按旧参数调用（gpt-3.5/4 / DeepSeek / Kimi）
kwargs_old = {
    "model": self.model, "messages": messages,
    "temperature": 0.2, "max_tokens": 800,
}
try:
    resp = client.chat.completions.create(**kwargs_old)
    return resp.choices[0].message.content
except Exception as exc:
    msg = str(exc)
    if "max_tokens" not in msg and "max_completion_tokens" not in msg:
        return ""    # 真正失败，直接返回

# 回退到新参数（o1 / o3 系列）
kwargs_new = {
    "model": self.model, "messages": messages,
    "temperature": 0.2, "max_completion_tokens": 800,
}
resp = client.chat.completions.create(**kwargs_new)
```

#### 兼容的 API 端点

| 提供方 | OPENAI_BASE_URL |
|---|---|
| OpenAI 官方 | (留空) |
| DeepSeek | `https://api.deepseek.com/v1` |
| Kimi (Moonshot) | `https://api.moonshot.cn/v1` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` |
| Ollama (本地) | `http://localhost:11434/v1` |

---

### 3.7 `app.py` — Streamlit 主程序

**核心问题**：把所有模块串起来 + 渲染双面板 + 处理用户交互。

#### 关键设计

1. **绝对路径解析**（避免 `streamlit run` 工作目录依赖）：
   ```python
   ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   _DEFAULT_TRIPLES = os.path.join(ROOT, "output", "triples_with_meta.csv")
   ```

2. **5 种缓存装饰器**（避免每个问题都重新构图）：
   ```python
   @st.cache_data    # 数据型：triples / entity_types / examples
   @st.cache_resource # 对象型：KGStore / Corpus / QueryParser
   ```

3. **侧栏可调参数**：
   - 数据源切换（triples CSV / 论文文本路径）
   - BFS 跳数（1~4）
   - 子图最大节点数（10~200）
   - 是否启用 LLM 重写
   - LLM 配置三件套（API Key / Base URL / Model）

4. **未识别实体兜底**：
   ```python
   if seed not in kg.graph:
       candidates = difflib.get_close_matches(seed, kg.all_entities(), n=6, cutoff=0.4)
       # 渲染按钮，点击后填入问题框重跑
   ```

5. **示例问题快速复用**（25 个样例，覆盖 8 种意图）

---

## 4. 端到端流程举例

以问题 **"全球鹰的速度是多少？"** 为例，走一遍完整流程：

### 4.1 用户输入 → 解析

```
question = "全球鹰的速度是多少？"
↓ QueryParser.parse()
  1. 正则匹配 → QueryIntent.VALUE, target_relation="has_value"
  2. NER 抽取 → mentions = [全球鹰, 速度]
  3. 捕获组置顶 → entities = [全球鹰, 速度]
↓
ParsedQuery(
    raw="全球鹰的速度是多少",
    intent=QueryIntent.VALUE,
    entities=["全球鹰", "速度"],
    target_relation="has_value",
    primary="全球鹰",
    secondary="速度"
)
```

### 4.2 检索

```python
retrieve(parsed, kg, corpus):
    intent = VALUE
    primary = "全球鹰"

    # 查 VALUE 相关关系
    for rel in ("has_value", "greater_than_value", "less_than_value", ...):
        rows.extend(kg.out_edges("全球鹰", rel))

    rows = [
        TripleRow(head="全球鹰", relation="has_value", tail="18000m", score=0.78, source="numeric"),
        TripleRow(head="全球鹰", relation="has_value", tail="740km/h", score=0.78, source="numeric"),
    ]

    # 原文证据
    evidence_rows = corpus.search("全球鹰", limit=6)

    # 辅助"更多相关"
    related_triples = [其他 全球鹰 的三元组][:10]

    return RetrievalResult(triples=rows, evidence_rows=..., subgraph_seed="全球鹰", ...)
```

### 4.3 模板答案

```
关于「全球鹰」的相关数值/取值信息：
  • (全球鹰 —has_value→ 18000m)
  • (全球鹰 —has_value→ 740km/h)

原文佐证：
  · [4.2节] 全球鹰的最大飞行高度可达 18000m，最大速度为 740km/h。
  · [4.1节] 诺斯罗普·格鲁曼公司研制的 RQ-4A 全球鹰是一种 HALE 长航时无人机。
```

### 4.4 子图可视化

- 中心节点：`全球鹰`（金色边框，34 号字）
- 主答案端点：`18000m`、`740km/h`（红色边框，24 号字）
- 主答案边：金色高亮加粗
- BFS 2 跳邻居：`无人机`、`美国`、`HALE`、`诺斯罗普·格鲁曼`、`升力`、`续航时间`...
- 普通邻居：灰色边框

---

## 5. 知识图谱可视化方案

### 5.1 中心子图构建

```python
def _build_visual_graph(kg, seed, highlight_triples, hops, max_nodes):
    g = kg.neighbors_subgraph(seed, hops, max_nodes)
    # 主答案补强：超出 hops 也强制加入
    for t in highlight_triples:
        if t.head not in g: g.add_node(t.head)
        if t.tail not in g: g.add_node(t.tail)
        g.add_edge(t.head, t.tail, label=t.relation, _hl=True)
    return g
```

**关键设计**：BFS K 跳子图 + 主答案补强。主答案三元组的端点哪怕超出 BFS 跳数也强行展示，避免"回答里提到的实体在图里找不到"。

### 5.2 节点配色（14 种实体类型）

```python
_TYPE_COLOR = {
    "AIRCRAFT":             "#4a90e2",  # 蓝
    "WING_CONFIGURATION":   "#48b884",  # 绿
    "PARAMETER":            "#e8b339",  # 黄
    "AERODYNAMIC_CONCEPT":  "#ec8064",  # 橙红
    "STRUCTURAL_COMPONENT": "#9b7bd4",  # 紫
    "CONTROL_METHOD":       "#e36a7e",  # 红
    "PERFORMANCE_METRIC":   "#3eafa8",  # 青
    "ORGANIZATION":         "#7a8fa6",  # 灰蓝
    "PERSON":               "#e6925e",  # 橘
    "TECHNOLOGY":           "#5b8fd6",  # 蓝
    "MATERIAL":             "#5ba994",  # 墨绿
    "FLIGHT_PHASE":         "#d18752",  # 深橙
    "EQUATION":             "#9aa5b1",  # 灰
    "CONCEPT":              "#c97485",  # 暗红
    "_UNKNOWN_":            "#b8c1cc",  # 浅灰
}
```

### 5.3 边按关系语义分组配色（11 组）

| 语义组 | 颜色 | 包含关系 |
|---|---|---|
| 分类层级 | 靛紫 `#5b6ee1` | `instance_of`, `is_a` |
| 组成结构 | 蓝灰 `#4a89dc` | `has_part`, `part_of`, `connected_to`, `combines_with` |
| 因果影响 | 暖橙 `#e8804b` | `causes`, `leads_to`, `affects`, `generates`, `provides` |
| 性能改善 | 翠绿 `#37b87f` | `improves`, `enhances`, `satisfies`, `verifies` |
| 性能下降 | 珊瑚红 `#e85d75` | `reduces`, `solves` |
| 数值比较 | 金棕 `#c2a663` | `has_value`, `greater_than`, `less_than`, `equals_to`, `approximately` |
| 控制驱动 | 蓝绿 `#3eafa8` | `controls`, `drives`, `driven_by`, `actuated_by` |
| 使用应用 | 中性灰蓝 `#7a8fa6` | `used_for`, `uses_method`, `applied_to`, `needs`, `depends_on` |
| 开发研制 | 紫罗兰 `#9166cc` | `develops`, `manufactures`, `originates_from` |
| 位置变换 | 沙棕 `#b08968` | `located_at`, `contains`, `transforms_to` |
| 章节归属 | 浅灰 `#a0a8b3` | `discussed_in` |

### 5.4 视觉权重三档

| 节点 | 大小 | 边框 | 字号 |
|---|---|---|---|
| 中心实体 | 34px | 金色 `#ff9f43` 厚边框 | 17 |
| 主答案端点 | 24px | 暖橙 `#ffb84d` 中边框 | 14 |
| 普通节点 | 18px | 浅灰 `#d6dde6` 细边框 | 13 |

| 边 | 宽度 | 颜色 | 样式 |
|---|---|---|---|
| 主答案边 | 3.2px | 金色 `#ffb84d` | 实线 |
| 高 score (≥0.85) | 2.0px | 语义组色 | 实线 |
| 中 score (≥0.65) | 1.5px | 语义组色（透明 78%） | 实线 |
| 低 score | 1.0px | 灰色 | 实线 |
| 弱关系 | 同上 | 同上 | **虚线**（`affects`/`verifies`/`needs`/`discussed_in` 等） |

### 5.5 物理引擎参数

```javascript
"physics": {
  "stabilization": { "enabled": true, "iterations": 220, "fit": true },
  "barnesHut": {
    "gravitationalConstant": -9000,
    "centralGravity": 0.18,
    "springLength": 160,
    "springConstant": 0.035,
    "damping": 0.55,
    "avoidOverlap": 0.6
  }
}
```

调高 `damping`（阻尼）和 `avoidOverlap`（避免重叠），节点不再频繁抖动。

---

## 6. LLM 协同与降级策略

### 6.1 降级矩阵

| 状态 | UI 表现 | 答案来源 |
|---|---|---|
| 未配置 API Key | 侧栏显示"LLM 未配置 · 仅用模板答案"，复选框默认不勾 | 模板 |
| 已配置但用户没勾选 | 侧栏显示"LLM 已就绪"，复选框未勾 | 模板 |
| 已配置且勾选 | 模板 + LLM 重写 | LLM（失败回退模板） |
| LLM 调用失败 | 侧栏"上次 LLM 错误：xxx" + 答案下方"LLM 调用失败已降级模板：xxx" | 模板 |

### 6.2 LLM 受材料约束

Prompt 强制要求：

```
2) 仅使用「材料」中出现的信息；如材料不足，请明确说"图谱与论文未提供更多信息"。
4) 不要编造，不要发挥与材料无关的内容。
```

实测 DeepSeek / Kimi 等都能遵守这个约束，避免幻觉污染。

---

## 7. 缓存与性能优化

### 7.1 5 种缓存

```python
@st.cache_data    # 不可变数据
def _cached_triples(path):           # 三元组 list
def _cached_entity_types(path):      # 实体 → 类型映射
def _cached_examples(path):          # 示例问题列表

@st.cache_resource    # 共享对象
def _cached_kg(path):                # KGStore（含 networkx 图）
def _cached_corpus(path):            # Corpus（含 1042 句）
def _cached_parser(vocab_tuple):     # QueryParser（含 AC 自动机）
```

**效果**：第一次加载约 2 秒，之后问任何问题都 < 100ms（除 LLM 调用）。

### 7.2 缓存失效

- 用户在侧栏改三元组路径 → 自动失效（key 是 path）
- 用户改 BFS 跳数 / 节点数 → 不失效（这些是渲染参数，不影响图谱）

---

## 8. 启动与使用

### 8.1 基本启动

```bash
uv run streamlit run app_qa/app.py
```

默认访问 `http://localhost:8501`。

### 8.2 自定义端口 + headless 模式

```bash
uv run streamlit run app_qa/app.py --server.port 8511 --server.headless true
```

### 8.3 启用 LLM

```bash
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.deepseek.com/v1
set OPENAI_MODEL=deepseek-chat
uv run streamlit run app_qa/app.py
```

或在侧栏"LLM 配置"展开输入。

### 8.4 示例问题（25 条覆盖 8 种意图）

文件：`app_qa/samples/example_questions.txt`

```
什么是变体飞行器？               # DEFINITION
单-双折叠翼是什么？
有哪些控制方法？                 # LIST_INSTANCE
列举无人机型号
全球鹰的速度是多少？             # VALUE
翼展是多少？
变体飞行器有哪些参数？           # PROPERTY
机翼由什么组成？
什么减少阻力？                   # CAUSE
折叠翼会影响什么？               # EFFECT
反馈线性化在哪章讨论？           # CHAPTER
升力 和 阻力 的关系              # RELATION
机翼 和 气动效率 有什么关系
总结全球鹰                       # SUMMARY
介绍变体飞行器
与机翼相关的有哪些               # NEIGHBOR
```

---

## 9. 局限与未来优化

### 9.1 当前局限

1. **意图覆盖有限**：12 种意图无法覆盖所有问法，未识别意图默认走 NEIGHBOR（兜底但不够精准）
2. **实体识别依赖词典**：未登录实体会被 `difflib` 兜底但有时找不到合理候选
3. **多跳推理弱**：RELATION 模式只展开 3 跳路径，没有路径排序
4. **不支持多轮对话**：每个问题独立处理，无上下文记忆

### 9.2 优化方向

1. **意图扩展**：加入"比较"（X 比 Y 谁的性能好）、"时间"（X 在哪一年提出）等更多意图
2. **基于子图嵌入的相似实体推荐**：用 node2vec 或 GraphSAGE 训练实体向量，找语义相近实体
3. **路径排序**：用 score / 关系类型 / 路径长度综合打分，选出最有信息量的 K 条路径
4. **多轮对话**：记录会话历史，把上次的 primary 实体作为上下文锚点
5. **LLM 函数调用**：把 KG 查询接口暴露给 LLM，让它自己决定查什么（Agent 模式）
6. **节点点击切换中心**：支持 pyvis 节点点击事件，无需重新输入问题

---

> **配套数据**：基于 `output/triples_with_meta.csv`（1114 条三元组 / 401 实体）；论文原文 `aftcln.txt`（1042 句 / 14 章节）。
