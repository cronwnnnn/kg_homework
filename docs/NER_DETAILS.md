# 实体识别（NER）详解

> 配套实验报告使用。本文档详细讲解知识图谱系统中实体识别（Named Entity Recognition）的完整工作流、核心算法（Aho-Corasick 自动机）、辅助正则（数值/型号）、重叠消歧策略，以及 NER 在整个流水线中扮演的"地基"角色。

> **关联文档**：[ALGORITHM_DETAILS.md](ALGORITHM_DETAILS.md) 介绍 6 个抽取算法；本文专门讲它们共同依赖的 NER 层。

---

## 目录

- [1. NER 的完整工作流](#1-ner-的完整工作流)
- [2. 实体词典（NER 的"原料"）](#2-实体词典ner-的原料)
- [3. 核心引擎：Aho-Corasick 自动机](#3-核心引擎aho-corasick-自动机)
- [4. 辅助正则：数值与型号](#4-辅助正则数值与型号)
- [5. 重叠消歧（最关键的工程化细节）](#5-重叠消歧最关键的工程化细节)
- [6. 可选扩展机制](#6-可选扩展机制)
- [7. NER 在整个流水线中的下游用途](#7-ner-在整个流水线中的下游用途)
- [8. 性能与质量指标](#8-性能与质量指标)
- [9. 未来可优化方向](#9-未来可优化方向)

---

## 1. NER 的完整工作流

```
论文原文 aftcln.txt (177 KB)
    │
    ▼
┌──────────────────────────────────────────────┐
│   预处理 (preprocess.py)                      │
│   - 去 [1] 引用、压缩空白                     │
│   - 章节切分（第N章 / 数字.数字 小节）         │
│   - 句子切分（句号、问号、感叹号、分号）       │
│   → 438 段、1042 个句子                       │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│   实体词典准备 (ans.py / entities_by_type.json)│
│   - 14 种类型，1473 个领域实体                 │
│   - 例：AIRCRAFT=全球鹰/捕食者/RQ-4A...        │
│         PARAMETER=升力/阻力/翼展/航程...        │
│         WING_CONFIGURATION=折叠翼/可变后掠...   │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│   HybridNER 初始化 (ner.py)                   │
│   - vocab 排序：按长度降序                      │
│   - 构建 Aho-Corasick 自动机（预编译失败指针）  │
│   - 数值 / 型号正则编译                         │
└──────────────────────────────────────────────┘
                    │
            对每个句子 N 字调用 extract()
                    │
        ┌───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼
   ① 词典AC      ② 型号正则   ③ 数值正则   ④ (公式正则)
   _extract_vocab _extract_   _extract_     未启用
                  aircraft_code numeric
        │           │           │
        └───────────┼───────────┘
                    ▼
        ┌──────────────────────────┐
        │   重叠消歧                │
        │   _dedup_overlap()        │
        │   长 mention 优先         │
        │   同位置去重              │
        └──────────────────────────┘
                    ▼
              list[Mention]
       (text, start, end, etype)
```

每个句子产生一组 `Mention` 实体提及：

```python
@dataclass(frozen=True)
class Mention:
    text: str          # 实体文本
    start: int         # 起始位置
    end: int           # 结束位置
    etype: str         # "vocab" / "numeric" / "aircraft_code"
```

---

## 2. 实体词典（NER 的"原料"）

### 2.1 词典从哪来

`ans.py` 里维护了 `EntityLibrary` 类，把领域知识按 14 类组织。运行 `tools/export_entities_by_type_json.py` 后导出到 `data/entities_by_type.json`：

```json
{
  "AIRCRAFT": ["全球鹰", "捕食者", "X-47B", "F-22", "RQ-4A", ...],
  "WING_CONFIGURATION": ["折叠翼", "可变后掠翼", "倾转旋翼", "双翼", ...],
  "PARAMETER": ["升力", "阻力", "翼展", "航程", "升阻比", ...],
  "AERODYNAMIC_CONCEPT": ["失速", "湍流", "层流", "雷诺数", ...],
  "STRUCTURAL_COMPONENT": ["机翼", "机身", "起落架", "尾翼", ...],
  "CONTROL_METHOD": ["反馈线性化", "PID 控制", "滑模控制", ...],
  "PERFORMANCE_METRIC": ["巡航高度", "最大速度", "续航时间", ...],
  "ORGANIZATION": ["NASA", "波音", "诺斯罗普·格鲁曼", "清华大学", ...],
  "PERSON": ["郭廷宇", "李航", ...],
  "TECHNOLOGY": ["复合材料制造", "电传飞控", ...],
  "MATERIAL": ["铝合金", "碳纤维", "钛合金", "复合材料", ...],
  "FLIGHT_PHASE": ["巡航", "起飞", "降落", "悬停", ...],
  "EQUATION": ["纳维-斯托克斯方程", "贝努利方程", ...],
  "CONCEPT": ["变体", "气动外形", "操纵性", ...]
}
```

**14 类总计 1473 个实体**（去重后），这是 NER 词典的"底座"。

### 2.2 词典在系统里的两层身份

| 身份 | 用途 | 文件 |
|---|---|---|
| **flat_vocab**（扁平词表） | 喂给 HybridNER 做 AC 自动机匹配 | `run_extract.py: load_domain_vocab()` |
| **entities_by_type**（按类型分组） | 喂给 TypeBasedExtractor 打 `instance_of` 标签 | 同上 |

### 2.3 词典预过滤（噪声治理）

构建词典时**预过滤噪声**：

```python
_EXCLUDED_VOCAB_ETYPES = {"NUMERIC_VALUE"}  # 数值不当实体
_EXCLUDED_VOCAB_TERMS = {
    "是", "为", "在", "有", "无", "其", "之", "者",          # 过短虚词
    "部分", "情况", "状态", "方式", "方法", "需要", "需求",  # 无信息量
    "影响", "决定", "可能", "进而", "因此", "通过", "使用",
    "研究", "本文", "本章", "公式", "图表", ...
}
```

> 不过滤这些会导致 NER 在每句话里都误命中"是""影响"等，把 trigger_extractor 的 score 算崩。

---

## 3. 核心引擎：Aho-Corasick 自动机

这是 NER 性能的关键。下面先讲为什么用它，再讲怎么用。

### 3.1 朴素方法的问题

如果用最朴素的方法（每个实体在每句话里 `text.find(word)`），时间复杂度是：

```
T = O(句子数 × 词典大小 × 每次 find 复杂度)
  = O(1042 × 1473 × 平均句长 60)
  = O(92,000,000)  ≈ 9200 万次字符比较
```

实测大概要 30 秒以上才能跑完。

### 3.2 AC 自动机的妙处

**Aho-Corasick** 是 1975 年提出的**多模式串匹配**算法。它的核心思想：

1. **预编译**：把所有模式串构造成一棵 **Trie 树**
2. **加失败指针**：每个节点都有一条"匹配失败时跳到哪里继续"的指针（类似 KMP 的 next 数组）
3. **匹配时**：对文本只扫一遍，遇到失配就沿失败指针跳，能同时检测所有模式

时间复杂度：**O(N + 命中数)**，N 为文本长度，与词典大小无关。

```
T_AC = O(1042 × 60 + 命中数)
     ≈ O(62,520 + 数百次命中)
     ≈ 几万次操作
```

**性能提升约 1000 倍**，实测 1042 句全部跑完只要 1~2 秒。

### 3.3 代码实现

`ner.py` 中初始化：

```python
self._automaton = ahocorasick.Automaton()
for idx, word in enumerate(self._vocab):
    if len(word) >= 2:                              # 单字过滤
        self._automaton.add_word(word, (idx, word))
self._automaton.make_automaton()                    # 预编译失败指针
```

匹配时一次扫描出所有命中：

```python
def _extract_vocab(self, text: str) -> list[Mention]:
    out = []
    for end_index, (_, word) in self._automaton.iter(text):
        start = end_index - len(word) + 1
        out.append(Mention(text=word, start=start,
                          end=end_index + 1, etype="vocab"))
    return out
```

### 3.4 降级方案

万一 `pyahocorasick` 没装，代码会优雅降级到朴素 `text.find` 循环：

```python
try:
    import ahocorasick
except ImportError:
    ahocorasick = None
```

慢，但能跑（适合不能装 C 扩展的环境）。

---

## 4. 辅助正则：数值与型号

### 4.1 数值正则

**为什么单独做**：词典里没法穷举所有数值（"400km/h"、"18000m"、"0.8Mach"...），而且数值实体有强格式特征，正则比词典更合适。

**三类模式**：

```python
_NUMERIC_PATTERNS = [
    # ① 带单位数值（30+ 单位白名单）
    re.compile(r"""\d+(?:\.\d+)?\s*(?:
        km/h|m/s|kt|kg|t|g|kw|kW|m|cm|mm|s|ms|km|kn|
        N·m|N·m|Nm|N|度|°|°C|%|Mach|马赫|克|吨|
        公斤|公里|米|秒|赫兹|Hz|分钟|小时|h
    )"""),

    # ② 科学计数法
    re.compile(r"\d+(?:\.\d+)?[×x]\d+(?:\.\d+)?\^?\d+"),

    # ③ 数值区间
    re.compile(r"\d+\s*-\s*\d+\s*(?:km/h|m/s|kg|t|m|cm|mm|°|度|%|...)")
]
```

**实例**：

句子：`"全球鹰的最大飞行高度可达 18000m，最大速度为 740km/h。"`

- 正则 ① 命中：`18000m`、`740km/h`
- 输出两个 `Mention`：
  ```python
  Mention("18000m", start=14, end=20, etype="numeric")
  Mention("740km/h", start=27, end=34, etype="numeric")
  ```

这些数值会作为 `numeric_extractor` 抽取 `has_value` 关系的 tail。

### 4.2 型号正则

**为什么单独做**：军事/民用飞行器型号有固定格式：**字母前缀-数字-可选后缀**。词典又难以穷举所有型号变体。

**正则**：

```python
_AIRCRAFT_CODE_RE = re.compile(r"\b[A-Z]{1,3}-?\d{1,3}[A-Z]?\b")
```

**匹配示例**：

- `RQ-4A`（无人机型号）
- `MQ-9`（捕食者）
- `F-111`（战斗机）
- `X-47B`（实验机）
- `B-1`（轰炸机）

`\b` 是单词边界，避免误抓嵌在中文里的 "A1" 这类无关字符串。

**实例**：

句子：`"诺斯罗普·格鲁曼公司研制的 RQ-4A 全球鹰是一种 HALE 长航时无人机。"`

- 正则命中：`RQ-4A`
- 输出 `Mention("RQ-4A", start=15, end=20, etype="aircraft_code")`

这条 mention 哪怕词典没有 "RQ-4A"，也能进入下游 trigger 抽取流程。

---

## 5. 重叠消歧（最关键的工程化细节）

### 5.1 为什么需要

不同算法可能给同一段文本产生重叠 mention：

```
文本：           "折叠翼变体飞行器"
词典 AC 命中：     [折叠翼]
                  [变体飞行器]
                  [折叠翼变体飞行器]   ← 长复合词
```

如果不消歧，下游 trigger_extractor 会把"折叠翼"和"变体飞行器"也当成独立端点，产生大量冗余三元组。

### 5.2 消歧规则

```python
@staticmethod
def _dedup_overlap(mentions):
    # 排序：起始位置升序 + 同位置长度降序
    mentions = sorted(mentions, key=lambda m: (m.start, -(m.end - m.start)))
    kept = []
    for m in mentions:
        # 规则1：与上一个 mention 重叠且自己更短 → 跳过
        if kept and m.start < kept[-1].end and (m.end - m.start) <= (kept[-1].end - kept[-1].start):
            continue
        # 规则2：与上一个 mention 范围完全相同 → 跳过
        if kept and m.start == kept[-1].start and m.end == kept[-1].end:
            continue
        kept.append(m)
    return sorted(kept, key=lambda x: x.start)
```

**核心准则**：**长 mention 优先 + 同位置去重**。

### 5.3 实例

输入 mentions：

```
Mention("折叠翼",          start=0, end=3)
Mention("折叠翼变体飞行器",  start=0, end=9)
Mention("变体飞行器",      start=3, end=9)
Mention("飞行器",          start=6, end=9)
```

排序后（按 start 升序，同位置按 length 降序）：

```
("折叠翼变体飞行器", 0, 9)    ← 最长，保留
("折叠翼",          0, 3)    ← 与上一个重叠且短，跳过
("变体飞行器",      3, 9)    ← 与上一个重叠（被包含）且短，跳过
("飞行器",          6, 9)    ← 同上，跳过
```

**输出**：只保留 `折叠翼变体飞行器`。

### 5.4 副作用与权衡

这条规则也带来一个**已知问题**：长复合实体会"吞并"短实体（README §10.2 提到这是回滚 LLM NER 词典扩充的原因）。比如：

- 词典加了长词 `"前后折叠翼"` 后，原本独立的 `"折叠翼"` 会被吞没
- 评估时金标只标了 `"折叠翼"`，pred 输出 `"前后折叠翼"`，导致 FN +1

**解决思路**（README 提到的 `tools/archive/prune_long_ner.py`）：

- 把会"覆盖式吞并"短实体的过长复合词从词典里剔除

---

## 6. 可选扩展机制

### 6.1 动态加词

`HybridNER.add_terms()` 支持运行时往词典加新词：

```python
def add_terms(self, terms):
    new_terms = [t for t in terms if t not in self._vocab_set and len(t) >= 2]
    if not new_terms:
        return
    self._vocab.extend(new_terms)
    self._vocab_set.update(new_terms)
    # 重建 AC 自动机
    self._automaton = ahocorasick.Automaton()
    for idx, word in enumerate(self._vocab):
        self._automaton.add_word(word, (idx, word))
    self._automaton.make_automaton()
```

这条接口曾给已归档的 `paper_entity_recognizer.py` 使用，spaCy 挖掘出来的"论文内出现但词典没收"的实体可以并入词典。

### 6.2 当前禁用的扩展

| 模块 | 状态 | 原因 |
|---|---|---|
| `extractors/archive/paper_entity_recognizer.py`（spaCy 实体挖掘） | 归档 | 长复合词吞并短实体导致 F1 下降 |
| `tools/archive/llm_ner_expand.py`（DeepSeek 全文 NER） | 归档 | LLM 提议 191 词，合并 159 后 F1 反而下降，已回滚 |

---

## 7. NER 在整个流水线中的下游用途

| 抽取器 | 怎样用 NER |
|---|---|
| `TriggerCooccurrenceExtractor` | 句中实体两两配对，扫描 gap 找触发词 |
| `PatternExtractor` | 触发词左右窗口查 right_most / left_most 实体 |
| `NumericExtractor` | raw head 通过词典子串回填到 canonical |
| `DependencyREExtractor` | `canonicalize()` 把依存子树规约到词典实体 |
| `TypeBasedExtractor` | "observed entities" 集合 ∩ 词典 → `instance_of` 关系 |
| `ChapterMembershipExtractor` | 实体出现频次统计 → `discussed_in` |
| `app_qa.query_parser` | 用户问题里的实体识别（共用 HybridNER） |

**也就是说，整个系统的所有关系都建立在 NER 命中的实体之上**。NER 漏识别一个实体，整套关系都会缺失。

---

## 8. 性能与质量指标

| 指标 | 数值 |
|---|---|
| 词典实体数 | 1473（14 类） |
| 单次 NER 速度（1042 句） | ~1-2 秒 |
| 实际抽取实体数 | 401（在 pred 三元组中出现的） |
| 实体级 P / R / F1 | 0.9664 / 0.5308 / **0.6852** |

**Precision 几乎完美（0.97）**：识别到的实体绝大多数正确。**Recall 较低（0.53）** 是当前 F1 瓶颈：

- 论文里有真实出现但词典没收录的实体（如长复合 NP "上下机翼相互干扰"）
- 金标里包含一些组合型实体（如 "公转角度_单翼状态" → 词典只有 "公转角度"）

---

## 9. 未来可优化方向

1. **重新启用 spaCy 论文实体挖掘**，但限制最大长度防吞并
2. **基于上下文的模糊匹配**：对 OOV 实体，允许 80% 字面相似度的词典词作 canonical
3. **半自动词典扩充**：用 LLM 离线挖掘新实体，但走 `audit_llm_ner.py` 严格审计
4. **细化实体类型**：当前 14 类不够细，比如 PARAMETER 既包含"升力"也包含"航程"，可拆 5~6 类
5. **加入字符级 fallback**：用 BPE/分词器对 OOV 段做切分，部分匹配也算半个命中
6. **实体类型一致性约束**：同一实体在不同句子可能被打多种类型，可基于章节上下文投票决定主类型

---

> **配套实验数据**：基于 `aftcln.txt`（清华大学郭廷宇博士论文）抽取 1114 条三元组 / 401 实体；第四章金标 `gold/gold_triples_augmented.csv` 745 条，最新评估严格 F1=0.5752、实体级 F1=0.6852。
