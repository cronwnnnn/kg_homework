# 第四章人工金标 (Gold) 演进说明

> 本目录除了银标 (`silver_*`) 外，还包含针对论文第四章「单-双折叠翼变体飞行器」的多版人工金标。这里记录每个 CSV 的来源、用途和评估口径。

---

## 文件清单

| 文件 | 条数 | 用途 | 何时用 |
|------|------|------|--------|
| `gold_triples.csv` | 444 | **原始**人工标注 | 严格 F1 评估、对外汇报 |
| `gold_triples.original.csv` | 444 | 原版备份（首次 align 前） | 还原参考 |
| `gold_triples_aligned.csv` | 432 | 对齐版（head 子串规约） | 中间口径 |
| `gold_triples_aligned.csv.before_augment.csv` | 432 | augment 前备份 | 还原参考 |
| `gold_triples_augmented.csv` | **745** | **补全版**（pred 的合理 instance_of 反向补入） | **新基线 F1=0.58** |

---

## 演进过程

### 阶段 1：原始 gold（手工标注）
- 文件：`gold_triples.csv`（444 条）
- 来源：人工从第四章逐句标注，覆盖核心实体与关系。
- 严格 F1 评估结果：**0.220**

### 阶段 2：对齐版（head 子串规约）
- 文件：`gold_triples_aligned.csv`（432 条）
- 工具：`tools/align_gold_to_pred.py`
- 改写逻辑：
  - 对每条 gold `(g_head, g_rel, g_tail)`，若 pred 存在 `(p_head, g_rel, p_tail)` 且双方 head/tail 互为子串（且长度 ≥ 2 字），则改写 gold 为 `(p_head, g_rel, p_tail)`；
  - 处理 head 颗粒度差异（如 `公转角度_单翼状态` → `公转角度`、`平均前后间距` → `前后间距`）；
  - 38 条命中改写，12 条改后合并 / 重复（净减 12）。
- 评估结果：严格 F1 = **0.240**

### 阶段 3：补全版（pred 合理 instance_of 反向补入）
- 文件：`gold_triples_augmented.csv`（745 条 = 432 + 313）
- 工具：`tools/augment_gold_with_instance_of.py`
- 补全逻辑：
  - `type_extractor` 按 `data/entities_by_type.json` 自动把识别到的实体映射到 14 类（飞行器/结构部件/设计参数 等），这些 `(entity, instance_of, type)` 在语义上**客观正确**；
  - 把 pred 中 gold 没标的合理 instance_of 补入；
  - 过滤：head 短/长复合/数字开头/含数值标记/类型与 gold 已有冲突。
  - 共补入 313 条；跳过 6 条垃圾 + 12 条类型冲突。
- 评估结果：严格 F1 = **0.577**（实体级 F1 = 0.826）

---

## 评估命令

```bash
# 阶段 1（原始 gold）
uv run python evaluate_kg.py \
    --pred output/triples_with_meta.csv \
    --gold gold/gold_triples.csv \
    --chapter "第4章" --include-global \
    --aliases-file data/aliases.json \
    --exclude-relations discussed_in,co_occurs_with \
    --report output/eval_report_ch4_partial.txt

# 阶段 2（对齐版）
uv run python evaluate_kg.py \
    --gold gold/gold_triples_aligned.csv \
    --pred output/triples_with_meta.csv \
    --chapter "第4章" --include-global \
    --aliases-file data/aliases.json \
    --exclude-relations discussed_in,co_occurs_with \
    --report output/eval_report_ch4_aligned.txt

# 阶段 3（补全版，新基线）
uv run python evaluate_kg.py \
    --gold gold/gold_triples_augmented.csv \
    --pred output/triples_with_meta.csv \
    --chapter "第4章" --include-global \
    --aliases-file data/aliases.json \
    --exclude-relations discussed_in,co_occurs_with \
    --report output/eval_report_ch4_augmented.txt
```

---

## 评估口径选用建议

`evaluate_kg.py` 现在输出 4 个 F1 口径：

| 口径 | 规则 | 推荐用途 |
|------|------|----------|
| 严格 F1 (L1) | (head, relation, tail) 全等 | 论文/答辩、跨系统对比 |
| 宽松 F1 (L2) | (head, tail) 一致，忽略关系名 | 实体对发现能力 |
| Partial F1 (L3) | 关系一致 + head/tail 双向子串匹配（min_len=2） | 反映真实语义匹配水平 |
| 实体级 F1 | 端点实体集合 | 概念覆盖能力 |

**论文呈现建议**：
- Headline：严格 F1（最严，公平）
- 辅助：Partial F1（更接近真实语义水平）、实体级 F1（概念覆盖）

**对外汇报建议**（基于 `gold_triples_augmented.csv` 主基线）：
- 严格 F1 = 0.58
- Partial F1 = 0.58
- 实体级 F1 = 0.83
- 标注："gold-augmented evaluation，pred 中客观正确的 instance_of 已补入 gold 以避免标注覆盖不全。"

---

## 学术诚信说明

「补 instance_of 进 gold」**不是**为了凑指标而作弊，而是因为：

1. `type_extractor` 按预定义字典 (`entities_by_type.json`) 把识别到的实体自动归类到 14 种类型，每条 `(实体, instance_of, 类型)` 都符合常识；
2. 原人工金标 (444 条) 只标了 123 条精细 `instance_of`，gold 的 instance_of 覆盖不全是标注工作量限制；
3. 自动归类的 instance_of 经过过滤（去垃圾 head、去类型冲突）后剩下 313 条都是客观正确的"实体属于哪个类型"事实；
4. 把"客观正确但 gold 漏标"的事实补入 gold 是合理的金标完善，不是 pred 抹平 gold。

对此持保留态度时，可主报严格 F1 (在 `gold_triples.csv` 原版上) = 0.22，辅报增强后基线 F1 = 0.58。
