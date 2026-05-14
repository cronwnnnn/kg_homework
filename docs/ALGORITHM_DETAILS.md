# 关系抽取算法原理详解

> 配套实验报告使用。本文档详细讲解单-双折叠翼变体飞行器知识图谱构建系统中使用的 6 类自研抽取算法（含 1 个 LLM 增强层）的工作原理、关键设计、置信度计算与各自的优缺点。

> **配套数据**：基于 `aftcln.txt`（清华大学郭廷宇博士论文，177 KB）抽取 1114 条三元组 / 401 实体；第四章金标 `gold/gold_triples_augmented.csv` 745 条，最新评估严格 F1=0.5752、Partial F1=0.5962、实体级 F1=0.6852、`instance_of` F1=0.799。

---

## 目录

- [0. 整体流程与多算法协同框架](#0-整体流程与多算法协同框架)
- [1. 预处理与命名实体识别（NER）](#1-预处理与命名实体识别ner)
- [2. 算法一：触发词共现抽取 TriggerCooccurrenceExtractor（主力之一）](#2-算法一触发词共现抽取-triggercooccurrenceextractor主力之一)
- [3. 算法二：高精度模板抽取 PatternExtractor](#3-算法二高精度模板抽取-patternextractor)
- [4. 算法三：数值关系抽取 NumericExtractor](#4-算法三数值关系抽取-numericextractor)
- [5. 算法四：依存句法增强抽取 DependencyREExtractor](#5-算法四依存句法增强抽取-dependencyreextractordep)
- [6. 算法五：类型归属抽取 TypeBasedExtractor（F1 主力，贡献 94% TP）](#6-算法五类型归属抽取-typebasedextractorf1-主力贡献-94-tp)
- [7. 算法六：章节归属抽取 ChapterMembershipExtractor（章节问答用）](#7-算法六章节归属抽取-chaptermembershipextractor章节问答用)
- [8. 关系归一化 RelationNormalizer](#8-关系归一化-relationnormalizer)
- [9. LLM 增强层 LLMEnhancer（可选）](#9-llm-增强层-llmenhancer可选)
- [10. 算法 P/R/F1 贡献汇总](#10-算法-prf1-贡献汇总最新评估)
- [11. 系统设计哲学（创新点）](#11-系统设计哲学创新点)

---

## 0. 整体流程与多算法协同框架

系统对论文 `aftcln.txt` 共采用 **6 类算法 + 1 个 LLM 增强层** 协同抽取知识三元组。每条三元组都带 `source` 字段标明来源算法，便于消融与回溯。

```
       ┌───────────────────────────────────────────────────┐
       │            预处理 TextPreprocessor                 │
       │  按行扫描 → 维护章节 / 小节 → 段落切分 → 句子切分 │
       │  正则：第N章 / N.M / 中文句末标点（。！？；）       │
       │  输出：438 段、1042 句（携带 chapter / section）   │
       └────────────────────────┬──────────────────────────┘
                                │
       ┌────────────────────────▼──────────────────────────┐
       │   混合实体识别 HybridNER（AC自动机 + 双正则）      │
       │   - 词典 1473 词（AC自动机 O(N) 多模式匹配）        │
       │   - 数值正则（400km/h / 12t / 9000N·m / 25%）       │
       │   - 型号正则（RQ-4A / MQ-9 / F-111）               │
       │   长 mention 优先 + 同范围去重                      │
       └────────────────────────┬──────────────────────────┘
                                │
   ┌─────┬─────────┬────────────┼────────────┬───────────┐
   ▼     ▼         ▼            ▼            ▼           ▼
  ① 触发词  ② 模板    ③ 数值       ④ 依存      ⑤ 类型     ⑥ 章节
  共现     (高精度)  关系         增强        归属       归属
  trigger  pattern   numeric      dep         type      chapter
   │         │         │           │           │          │
   └──────────────合并 + 去重────────────────┘          │
                       │                                  │
              ┌────────▼─────────┐                        │
              │ LLM 增强层（可选）│                       │
              │ - mock：去重清洗  │                       │
              │ - polish：质检     │                       │
              │ - discover：发现   │                       │
              └────────┬─────────┘                        │
                       │                                  │
              ┌────────▼─────────┐                        │
              │ 关系归一化         │                       │
              │ RelationNormalizer │                       │
              │ 33 种统一关系本体  │                       │
              └────────┬─────────┘                        │
                       │                                  │
              ┌────────▼─────────┐                        │
              │ 全局 score 阈值过滤 (≥0.55)                │
              └────────┬─────────┘                        │
                       ▼                                  │
            最终三元组 (head, relation, tail, score, source)
```

**协同设计要点**：

1. **多算法独立产出，统一去重融合**：避免单一算法天花板（任何一种方法都有盲区）。
2. **置信度分级**：模板 0.82–0.85 > 类型 0.92 > 数值 0.78 > 依存 0.66–0.78 > 触发词 0.40–1.00 > 章节 0.55–0.80。
3. **`source` 字段全程传递**：评估时可按来源做消融分析（README §12.3 给出了各抽取器对最终 F1=0.6949 的具体贡献）。
4. **可降级**：spaCy 缺失时依存抽取静默跳过；LLM 无 API 时降为 mock 去重，不破坏流水线。

---

## 1. 预处理与命名实体识别（NER）

### 1.1 文本预处理 `preprocess.py`

**核心问题**：把整本中文论文切成"句子+章节归属"的结构化记录，为后续句级抽取提供原料。

**算法**：

| 步骤 | 正则 / 规则 | 作用 |
|---|---|---|
| 清洗 | 删除 `[1]` / `[1-3]` 文献引用、压缩空白 | 降噪 |
| 章节识别 | `^(第[一二三四五六七八九十0-9]+章\s*[^\n]{0,40})$` | 维护 `current_chapter` |
| 小节识别 | `^(\d+(?:\.\d+){1,3}\s+[^\n]{0,60})$` | 维护 `current_section`（如"4.2.1"） |
| 句子切分 | 后看断言 `(?<=[。！？；])\s*` | 中文句末四标点统一切句 |
| 长度过滤 | `min_sentence_len=6` | 去掉过短行（公式编号 / 表号） |

**产出**：每个句子记录 `(text, chapter, section, paragraph_id, sentence_id)`，最终 438 段 / 1042 句。

### 1.2 混合命名实体识别 `HybridNER`

**核心问题**：在每个句子里高效识别"领域实体提及"（mention），作为关系抽取的端点。

**三路并行匹配**：

#### (1) 词典匹配 — Aho-Corasick 自动机

- 输入：1473 个领域词典实体（来自 `ans.py`）
- 构建：把所有词典词加入 AC 自动机，预编译失败指针
- 匹配：对每个句子做一次 `automaton.iter(text)`，时间复杂度 **O(N + 命中数)**
- 相比逐个 `find`：从 O(N·M·L) 降到 O(N)，是支撑 1042 句快速 NER 的关键

```python
self._automaton = ahocorasick.Automaton()
for idx, word in enumerate(self._vocab):
    if len(word) >= 2:
        self._automaton.add_word(word, (idx, word))
self._automaton.make_automaton()
for end_idx, (_, word) in self._automaton.iter(text):
    yield Mention(word, start=end_idx-len(word)+1, end=end_idx+1, etype="vocab")
```

#### (2) 数值正则

三类模式：
- 带单位数值：`\d+(?:\.\d+)?\s*(km/h|m/s|kg|t|...|Mach|马赫|...)` → 匹配 `400km/h`、`12t`
- 科学计数：`\d+[×x]\d+\^?\d+` → 匹配 `7×10^5`
- 数值区间：`\d+\s*-\s*\d+\s*单位` → 匹配 `5-10kg`

#### (3) 型号正则

`\b[A-Z]{1,3}-?\d{1,3}[A-Z]?\b` → 匹配 `RQ-4A`、`MQ-9`、`F-111`、`X-47B`

#### 重叠消歧（关键）

```python
mentions = sorted(mentions, key=lambda m: (m.start, -(m.end - m.start)))
for m in mentions:
    if 与已保留 mention 重叠 且 自己长度更短: skip
    if 与已保留 mention 范围完全相同: skip
```

**规则**：长 mention 优先 + 同位置去重。
- 例：句中"折叠翼变体飞行器"，若词典同时收了"折叠翼"、"变体飞行器"、"折叠翼变体飞行器"，只保留最长者。
- 避免：短词重复匹配产生大量虚假端点。

---

## 2. 算法一：触发词共现抽取 `TriggerCooccurrenceExtractor`（主力之一）

**论文位置**：中文信息抽取最常用的强 baseline，本系统的"动词类"关系主要靠它。

### 2.1 算法思想

> 句子里识别出的实体两两配对，扫描两实体之间的窗口文本，若窗口内命中预定义触发词，则建立 `(头实体, 触发词→关系, 尾实体)` 三元组。

形式化：对句子 \(S\) 中实体提及序列 \(M = (m_1, m_2, ..., m_n)\)，对每对 \((m_i, m_j)\) (\(i < j\))，定义：
- **gap 窗口** \(g_{ij} = S[m_i.\text{end} : m_j.\text{start}]\)
- 若 \(g_{ij}\) 含触发词 \(t \in T\)，且 \(t\) 映射到关系 \(r = \phi(t)\)，则产出候选 \((m_i, r, m_j)\)
- 候选打分 \(\text{score}(g_{ij}, t, m_i, m_j) \geq 0.55\) 即保留

### 2.2 关键工程化约束（决定 P/R）

| 约束 | 代码 | 作用 |
|---|---|---|
| 仅相邻 K=6 个实体配对 | `j in range(i+1, min(n, i+1+K))` | 避免远距离误关联（O(n²) → O(n·K)） |
| `max_window=30` 字符 | `if len(gap) > 30: skip` | 触发词与实体相距过远视为无关 |
| 中间不跨句末标点 | `if any(c in "。！？；" for c in gap): skip` | 防止误抓跨句关系 |
| 中间有第三实体且窗口≤12 | `if third_inside and len(gap)<=12: skip` | 优先短窗口配对 |
| 否定情境降分 | gap 或前后 6 字符含 `不/未/无/非...` → -0.30 | 降低否定句误抽 |
| 被动语态翻转 | gap 前缀为"由"/"被" 且 rel∈{develops, manufactures} → 头尾对调 | 处理"由 X 研制" → X 是 head |
| 关系-类型约束 | `is_a` / `instance_of` 的 tail 禁止数值类型 | 防"X 是 25kg"被当成 is_a |

### 2.3 置信度评分（5 维加权）

```
base = 0.40 + min(len(trigger), 4) × 0.08          # 触发词越长越可信
    - 0.30 × (gap_extra / max_window)              # 距离越远越扣分
    + 0.05 × (len(head) ≥ 3)                       # 长实体加分
    + 0.05 × (len(tail) ≥ 3)
    - 0.30 × negated                                # 否定扣分
    + 0.20 / +0.10 / -0.15  按"触发词紧贴实体"分级    # X 触发词 Y 最佳
    - 0.25 × weak_trigger 且 prefix/suffix > 2       # 弱触发词远离实体扣重分
    - 0.10 × ("的"出现≥2 次)                          # 断句不准的信号
clip(score, 0.05, 1.0)
```

> 弱触发词指 `WEAK_TRIGGERS = {"为", "是", "属于", "影响", "决定", ...}`，这些短词歧义大，必须真正紧贴实体才计入。

### 2.4 触发词本体（33 种关系，160+ 触发词）

`RelationOntology.TRIGGER_TABLE` 一对多映射：

```python
"has_part":   ["由其组成", "包括", "包含", "构成", "组成", "分为", ...]
"is_a":       ["是一种", "是一类", "属于", "称为", ...]
"improves":   ["显著提升", "大幅提升", "改善", "增强", "强化"]
"reduces":    ["大幅降低", "削弱", "抑制", "缓解", "缩小", ...]
"causes":     ["导致", "造成", "引发", "致使"]
"controls":   ["控制", "调控", "调节", "调整", "操纵"]
"develops":   ["研发", "开发", "提出了", "提出", "设计了", "设计出", ...]
# ... 共 33 个关系类
```

匹配时 **长触发词优先 + 不重叠**（`consumed` 标记数组），避免"显著提升"被"提升"先吃掉。

### 2.5 实例

句子：`"反馈线性化控制方法显著提升了飞行器的稳定性。"`
- NER：`反馈线性化控制方法`、`飞行器`、`稳定性`
- 配对 (`反馈线性化控制方法`, `稳定性`)：gap = `"显著提升了飞行器的"`
  - 命中触发词 `显著提升` → `improves`
  - score ≈ 0.40 + 0.32 + 0 + 0.05 + 0.10 = **0.87**
- 输出：`(反馈线性化控制方法, improves, 稳定性, source=trigger, score=0.87)`

### 2.6 优缺点

**优点**：
- 实现简单，无依存模型依赖，对中文长句鲁棒
- 一句可产出多条三元组，覆盖率高
- 通过触发词表灵活扩展新关系

**缺点**：
- 触发词列表手工维护（曾因 `提高/扩大/引起/进而` 等过宽触发词产生大量 FP，README §6.1 中删除 13 个）
- 不分析句法结构，复杂嵌套（如"X 通过 Y 影响 Z"）容易错位
- 依赖 NER，未登录实体直接漏

**贡献**：抽取 169 条三元组（占总数 15%），第四章 F1 约 0.6×，是 trigger 来源 F1 的天花板（与 type_extractor 互补）。

---

## 3. 算法二：高精度模板抽取 `PatternExtractor`

**论文位置**：触发词共现的"显式句式版"，处理 `由X驱动`、`X 与 Y 连接` 这类固定结构。

### 3.1 算法思想

> 不依赖 NER 端点配对，而是先定位"触发词正则模板"，再向左右窗口寻找最贴合的词典实体作头/尾。

```
"由 [.{0,8}] 驱动"      → driven_by
"由 [.{0,12}] 组成"     → has_part   (允许列举式 tail)
"包括/包含/含有/分为"     → has_part   (允许列举式 tail)
"用于/适用于/应用于"       → used_for
"是一种"                  → is_a
"位于" / "安装(在|于)"    → located_at
"转变为/变体为/切换为"     → transforms_to
"研制/生产/制造"           → manufactures
"研发/开发"                → develops
"起源于"                   → originates_from
"与 [.{0,12}] 连接/对接/铰接/固连" → connected_to  (双实体型)
```

### 3.2 头/尾选择策略（关键）

- **head**：触发词左侧 12 字符窗口里**最右最长**的词典实体（最贴近的主语）
- **tail**：触发词右侧 16 字符窗口里**最左最长**的词典实体（紧跟的宾语）
- **列举式 tail**（`has_part` 触发）：按 `、，,;；` 切片，每片各取一个 left_most 实体

```python
def _right_most_in(fragment):
    # 返回 (rightmost_position, length, word) 三元组排序后最大者
    best = max([(pos+len(w), len(w), w) for w in vocab if w in fragment])
    return best[2]
```

### 3.3 实例

句子：`"该飞行器由折叠翼变体机构、机身和起落架组成。"`
- 命中正则 `"由 .{0,12} 组成"` → has_part，列举式
- head 窗口（左 12 字符）：`"该飞行器"` → `飞行器`
- tail 窗口列举：`折叠翼变体机构` / `机身` / `起落架` 各产一条
- 输出 3 条三元组：
  - `(飞行器, has_part, 折叠翼变体机构, source=pattern, score=0.82)`
  - `(飞行器, has_part, 机身, source=pattern, score=0.82)`
  - `(飞行器, has_part, 起落架, source=pattern, score=0.82)`

### 3.4 优缺点

**优点**：
- 模板高确定性，固定 score=0.82~0.85，**Precision 比触发词共现高 0.15**
- 双实体型 `X 与 Y 连接` 显式捕获连接关系，避免误抓"连接"动词
- 列举式 tail 一次可产多条 `has_part`，对论文中常见的"由 A、B、C 组成"句式友好

**缺点**：
- 覆盖窄（仅 14 个模板），漏召回多
- 模板硬编码，新句式需要人工添加

**贡献**：64 条三元组（占 5.7%），P≈0.7。

---

## 4. 算法三：数值关系抽取 `NumericExtractor`

**论文位置**：专攻 `has_value` / `greater_than_value` / `less_than_value` 三种数值关系。

### 4.1 算法思想

> 对句子做四种正则模板匹配，把"参数 + 数值 + 单位"组合识别出来；参数名通过词典回填规约到 canonical。

四组正则：

| 模式 | 关系 |
|---|---|
| `X (为\|是\|约为\|达到\|可达) N 单位` | `has_value` |
| `X 在 N 单位 (左右\|附近\|以上\|以下)` | `has_value` |
| `X (大于\|高于\|不低于\|至少) N 单位` | `greater_than_value` |
| `X (小于\|低于\|不超过\|至多) N 单位` | `less_than_value` |

单位白名单 30+：`km/h`、`m/s`、`kg`、`t`、`N·m`、`%`、`Mach`、`马赫`、`公里`、`米`、`Hz`、`Pa` ...

### 4.2 head 规约（关键）

raw head 是正则捕获的中文段，可能不在词典里（如 `"全球鹰飞行高度"`）。`_resolve()` 在词典里找：

1. 完全相等 → 直接返回
2. 否则按词典长度降序找**最长子串**（如 `"全球鹰飞行高度"` 找到 `"全球鹰"`）
3. 找不到 → 丢弃（避免产生未登录 head）

### 4.3 实例

句子：`"全球鹰的最大飞行高度可达18000m。"`
- 正则 `"X 可达 N 单位"` 命中
  - raw head = `"全球鹰的最大飞行高度"`
  - `_resolve()` 找到词典子串 `"全球鹰"`
  - value = `"18000m"`
- 输出：`(全球鹰, has_value, 18000m, source=numeric, score=0.78)`

### 4.4 优缺点

**优点**：
- Precision 高（0.78 固定分），P≈0.8
- 处理数值类关系比触发词共现更专业（不会把数字识别成普通实体）

**缺点**：
- 模板仅 4 个，对"X 大约是 N"、"X 范围是 A-B 单位" 等变体覆盖不全
- raw head 子串回填可能"截太短"（如 `"飞行高度"` 被回填到 `"高度"`）

**贡献**：43 条三元组，金标 `has_value` F1=0.13（金标里 has_value 51 条，本系统只命中 4 条 TP；漏召回主要因 head 颗粒度对不齐）。

---

## 5. 算法四：依存句法增强抽取 `DependencyREExtractor`（dep）

**论文位置**：相比传统 SVO 抽取的升级版，2026-05-14 默认开启。

### 5.1 算法思想

> 用 spaCy 中文依存模型（`zh_core_web_sm`）解析每个句子，**遍历所有 VERB 节点**（不仅是 ROOT），从依存树中提取 (主语, 谓词, 宾语) 三元组，再用关系归一化映射到本体。

依存关系约定：
- **subject 边**：`nsubj`（主语）、`top`（话题）、`nsubjpass`（被动主语）、`csubj`（从句主语）
- **object 边**：`dobj`（直宾）、`pobj`（介宾）、`iobj`（间宾）、`attr`（表语）、`ccomp`/`xcomp`（补语从句）

### 5.2 与旧 SVO 抽取器的 7 个核心改进

1. **不强求 head/tail 在词典内**：用子树拼接得到完整 NP（如 `"上下机翼相互干扰"`）
2. **子树清洗**：去掉 PUNCT / `的/了/着/过/地/得/之`、模态词 `会/能/可/应`、副词 ADV
3. **触发词驱动归一化**：用 `RelationOntology.TRIGGER_TABLE` 把谓词映射到关系本体
4. **cop 模式**：识别 "X 为 N 单位" 这类系动词结构 → `has_value`
5. **否定/被动检测**：自动翻转 head/tail 或降分
6. **泛 head 黑名单**：`研究内容/本研究/上述分析/结果显示` 等通用名词不能当 head
7. **NP 规整**：长度 [2, 25]，去前后助词

### 5.3 算法流程（简化伪代码）

```
for token in doc:
    if token.pos == VERB:
        rel, trig = normalizer.normalize(token.text)
        if rel == 'related_to': continue
        subjects = [c for c in token.children if c.dep in SUBJECT_DEPS]
        objects  = [c for c in token.children if c.dep in OBJECT_DEPS]
        if not subjects:                            # 没主语，往上爬
            for ancestor in token.ancestors:
                subjects = [c for c in ancestor.children if c.dep in SUBJECT_DEPS]
                if subjects: break
        for s in subjects:
            head = clean_subtree(s)                 # NP 清洗
            if not valid(head): continue
            for o in objects:
                tail = clean_subtree(o, allow_numeric=True)
                if not valid(tail): continue
                head = canonicalize(head)           # 词典子串规约 + 黑名单
                tail = canonicalize(tail)
                if 关系约束(rel, tail) 违反: continue
                if passive: head, tail = tail, head
                score = base(0.72)
                       + 0.04 × (head in vocab)
                       + 0.04 × (tail in vocab)
                       + 0.03 × (2 ≤ len(head),len(tail) ≤ 8)
                       - 0.10 × (len(head)>14 or len(tail)>14)
                       - 0.20 × negated
                if score ≥ 0.55: produce triple(source='dep')
    elif token.dep == 'cop':                        # "X 为 N 单位"
        从 head.children 取 subjects，head 为表语
        如果表语含数字 → rel = 'has_value' (score=0.72)
        否则 → rel = 'is_a' (score=0.66)
```

### 5.4 关键设计：`canonicalize` 与 `clean_subtree`

- **`clean_subtree`**：按 token 顺序拼接，过滤 `_DROP_TOKENS`、模态词、单独的副词；处理"的/了/着"前后缀
- **`canonicalize`**：
  1. 去掉介词前缀（`在/对/通过/采用` 等）
  2. 拒绝代词 `这/那/它/其/本节/本章`
  3. 拒绝泛 head `研究内容/本研究/上述分析/结果显示`
  4. 拒绝模糊后缀 `过程中$ / ^以上 / ^上述`
  5. 词典子串回填：若 NP 不在词典但有词典子串覆盖 ≥ NP 长度的一半，则替换为词典 canonical

### 5.5 实例

句子：`"作动器驱动折叠翼实现变体过程。"`
- spaCy 依存：`驱动` 为 VERB，`作动器` 为 nsubj，`折叠翼` 为 dobj
- 谓词归一化：`"驱动"` → `drives`
- 头尾：`作动器` / `折叠翼`
- 校验：均在词典内（+0.04+0.04），长度合适（+0.03） → score=0.83
- 输出：`(作动器, drives, 折叠翼, source=dep, score=0.83)`

### 5.6 优缺点

**优点**：
- 真正使用句法结构，能处理触发词共现解决不了的长距离依存
- cop 模式补充数值关系（弥补 numeric_extractor 模板覆盖不全）
- 子树清洗 + 词典回填，对未登录实体也能产出

**缺点**：
- 强依赖 spaCy 中文模型（缺失时静默跳过）
- 依存解析对中文长句仍有错误，复杂嵌套会跑偏
- 长 NP 容易包含修饰冗余，需要长度 ≤14 才不扣分

**贡献**：39 条三元组（占 3.5%），README §13 中提到对 `has_part` F1 +1。

---

## 6. 算法五：类型归属抽取 `TypeBasedExtractor`（F1 主力，贡献 94% TP）

**论文位置**：本系统最关键的"高 F1 单点贡献者"。

### 6.1 算法思想

> 对论文中实际出现的实体，按预先维护的"类型词典"（14 类）映射到类型标签，产出 `instance_of` 关系。

形式化：词典 \(D = \{(\text{type}_k, \{e_{k,1}, e_{k,2}, ...\})\}_{k=1}^{14}\)，论文中"被观察到的实体"集合 \(O\)，输出：

\[
T = \{(e, \text{instance\_of}, \text{label}(\text{type})) : e \in O, e \in D[\text{type}]\}
\]

14 类映射：

```
AIRCRAFT → 飞行器
WING_CONFIGURATION → 机翼构型
PARAMETER → 设计参数
AERODYNAMIC_CONCEPT → 气动概念
STRUCTURAL_COMPONENT → 结构部件
CONTROL_METHOD → 控制方法
PERFORMANCE_METRIC → 性能指标
ORGANIZATION → 组织机构
PERSON → 人物
TECHNOLOGY → 技术
MATERIAL → 材料
FLIGHT_PHASE → 飞行阶段
EQUATION → 公式
CONCEPT → 概念
```

### 6.2 观察集合 \(O\) 的两条来源

实体进入 \(O\) 必须**在论文里真实出现过**，避免"词典里有但论文没提"的实体也被打类型标签。\(O\) 由两路汇入：

1. **抽取器副产物**：触发词 / 模板 / 数值等抽取器命中的 head/tail 自动加入 \(O\)
2. **`collect_cooccurring_entities` 辅助函数**：对句子做 NER 后，**同句出现且类型对在白名单内**的实体也加入 \(O\)

白名单（8 对类型对，启发式）：

```
(AIRCRAFT, PARAMETER) / (AIRCRAFT, PERFORMANCE_METRIC) / (AIRCRAFT, WING_CONFIGURATION) /
(AIRCRAFT, MATERIAL) / (AIRCRAFT, STRUCTURAL_COMPONENT) /
(STRUCTURAL_COMPONENT, MATERIAL) / (PERSON, ORGANIZATION) / (TECHNOLOGY, AIRCRAFT)
```

> 这条副作用很关键：原 `CooccurrenceTypeExtractor` 输出的 score=0.50 triple 全被阈值砍掉，但"实体进入 observed" 这个副作用让 instance_of 召回 +6%。

### 6.3 实例

- 词典：`AIRCRAFT: [全球鹰, 捕食者, X-47B, ...]`
- 论文中触发词抽取器命中：`全球鹰`（head 或 tail）
- 评估：`全球鹰 ∈ O`，`全球鹰 ∈ D['AIRCRAFT']` → 输出 `(全球鹰, instance_of, 飞行器)`
- 固定 score=0.92

### 6.4 优缺点

**优点**：
- **F1=0.93**（最新评估 P=0.96, R=0.90），单一关系贡献最大
- 算法极简，14 个类型映射 + 词典查表，无歧义
- 词典可扩展，新加 1 个 `AIRCRAFT` 实体直接多 1 条 TP

**缺点**：
- 类型粒度由人工词典决定，跨类型实体（如某飞行器同时也是 TECHNOLOGY）只能选一类
- 完全依赖 NER 命中：词典里有但论文没出现的实体不会输出
- 不抽取实体之间的关系，只产生类型层级

**贡献**：345 条三元组（占 31%），第四章 instance_of 关系 **TP=317、F1=0.799**（当前），是评估分数的命脉。

---

## 7. 算法六：章节归属抽取 `ChapterMembershipExtractor`（章节问答用）

### 7.1 算法思想

> 实体在某章节出现频次 ≥ N 次时，建立 `(实体, discussed_in, 章节标题)` 关系，用于问答助手"X 在哪章讨论"功能。

### 7.2 关键约束

- **二级类型白名单 `SECONDARY_TYPES`**：仅对 11 种实体类型（AIRCRAFT, WING_CONFIGURATION, TECHNOLOGY, ...）建关系，**排除 NUMERIC_VALUE**（数值不绑章节）
- **`min_occur=2`**：必须在该章节出现 ≥ 2 次（剔除偶现噪声）
- **空章节过滤**：`if not chapter: continue`（防止前言/无章节段落被命中）

### 7.3 置信度评分

```
score = 0.55 + min(0.25, n × 0.02)    # n 为出现次数
        # n=2 → 0.59
        # n=10 → 0.75
        # n≥13 → 0.80（封顶）
```

### 7.4 实例

- 句子扫描后：`反馈线性化` 在 "第4章" 出现 8 次
- 输出：`(反馈线性化, discussed_in, 第4章, source=chapter, score=0.71)`

### 7.5 优缺点与贡献

**优点**：
- 给问答系统提供"章节归属"维度
- 出现频次直接编码进 score，可解释

**缺点**：
- 评估时被 `--exclude-relations discussed_in` 屏蔽（章节归属不算关系抽取核心指标）
- 对长论文章节数较少时区分度低

**贡献**：443 条三元组（占 39.8%），评估时被排除，**对 F1 无直接影响但对 app_qa 必不可少**。

---

## 8. 关系归一化 `RelationNormalizer`

> 桥接所有抽取器的"自由谓词"到统一关系本体。

### 8.1 算法

```python
class RelationNormalizer:
    def __init__(self):
        # 1. 精确字典：trigger → relation
        self._exact_map = {trig: rel for trig, rel in TRIGGER_TABLE}
        # 2. 子串列表（按长度降序）
        self._substring_map = [(trig, rel) for trig, rel in TRIGGER_TABLE.long_first()]

    def normalize(predicate):
        if predicate in _exact_map:        # 优先精确命中
            return _exact_map[predicate], predicate
        for trig, rel in _substring_map:   # 子串命中
            if trig in predicate:
                return rel, trig
        return "related_to", predicate     # 都不命中 → 兜底
```

### 8.2 使用场景

- spaCy 给出谓词 `"显著提升"` → 归一化为 `improves`
- LLM 给出 `"减小"` → 归一化为 `reduces`
- 触发词抽取器内部已用 `TRIGGER_TABLE` 直接映射，不再调用 normalizer

---

## 9. LLM 增强层 `LLMEnhancer`（可选）

**核心约束**：课程红线要求"不允许只用 LLM"，因此 LLM 仅作用于传统算法已识别的候选。

### 9.1 三种模式

| 模式 | 命令 | 作用 |
|---|---|---|
| **mock** | 默认 | 仅做去重 + 边界清洗，无 API |
| **openai polish** | `--llm openai` | 按段落送审，让 LLM 删冗 / 归一化 |
| **openai discover** | `--llm openai --no-llm --llm-discover` | 按章送整段正文，让 LLM 补充正文可逐字核对的新三元组 |

### 9.2 polish 模式（已知反向优化）

提示词让 LLM 做 3 件事：

1. 删错 / 删冗
2. 归一化 relation
3. 补充少量额外关系（每段 ≤ 2 条）

**实测**：F1 -0.041（polish 会改写 head/tail 颗粒度导致 36 条 instance_of TP 不匹配），故默认禁用。

### 9.3 discover 模式（当前推荐配置）

按章节把整段正文 + 已有候选发给 LLM，让 LLM 补充新三元组。**代码侧严格校验**：

```python
allowed_set = _allowed_relation_labels()   # 关系白名单
for item in llm_response['new_triples']:
    if item.relation not in allowed_set: continue     # 关系白名单校验
    if item.head not in body: continue                # head 必须正文子串
    if item.tail not in body: continue                # tail 必须正文子串
    if (h, r, t) in existing: continue                # 不许重复
```

**实测**：290 条 LLM 候选 → 28 条进 pred → 1 TP + 27 FP，F1 -0.014，但三元组数 +284、实体数 +175，整体图谱更"丰满"。

### 9.4 mock 模式工作流

```python
def _mock_enhance(candidates):
    seen = set()
    out = []
    for tri in candidates:
        h, t = tri.head.strip(), tri.tail.strip()
        if not h or not t or h == t: continue
        if (h, tri.relation, t) in seen: continue
        seen.add((h, tri.relation, t))
        out.append(tri)        # 保留各抽取器原有关系，不再二次归一化
    return out
```

---

## 10. 算法 P/R/F1 贡献汇总（最新评估）

> 数据来源：`output/eval_report.txt`（mock LLM · 含 dep 增强）

| 来源 | 输出数 | 占比 | Top 关系 | 关键贡献 |
|---|---|---|---|---|
| `type` | 345 | 31% | `instance_of` | **F1=0.799, TP=317**（主力） |
| `chapter` | 443 | 39.8% | `discussed_in` | 评估排除，但 QA 应用必需 |
| `trigger` | 169 | 15.2% | 30+ 关系 | 一句多产，召回主力 |
| `pattern` | 64 | 5.7% | `has_part`, `is_a` | 高精度 P≈0.8 |
| `numeric` | 43 | 3.9% | `has_value` | 数值类专责 |
| `dep` | 39 | 3.5% | 多种动词关系 | spaCy 依存补漏 |
| 合并 | **1114** | 100% | — | 严格 F1=0.5752 |

**消融建议（写报告时）**：

1. **关掉 type**：F1 暴跌到 0.05~0.10（约 90% TP 来自 instance_of）
2. **关掉 chapter**：F1 不变（评估时已排除）
3. **关掉 trigger**：F1 -0.05（损失多关系召回）
4. **关掉 pattern**：F1 -0.02（高 P 候选损失）
5. **关掉 dep**：F1 -0.01（少量 has_part 损失）
6. **关掉 numeric**：F1 -0.005（has_value TP 损失）

---

## 11. 系统设计哲学（创新点）

1. **多算法独立产出 + 统一融合**：不押注单一方法，每个算法在自己擅长的关系类型上发挥
2. **`source` 全程透明**：每条三元组都能追溯产生算法，支持消融实验和调试
3. **5 层置信度分级**：模板 > 类型 > 数值 > 依存 > 触发词 > 章节，融合时高分覆盖低分
4. **课程红线下的 LLM 使用**：LLM 仅作 polish + discover，且 discover 必须有正文字面证据
5. **可降级流水线**：spaCy / LLM / pyahocorasick 缺失都能优雅降级
6. **关系本体先行**：33 种统一关系 + 160+ 触发词，所有算法都映射到同一本体，便于评估
7. **观察集合 \(O\)**：分离"实体存在"与"实体被讨论"两个概念，避免词典覆盖偏差污染评估
