# tools/archive/ — 一次性脚本归档

> 这些脚本各自完成了某个特定阶段的工作，**产物已经合入主线**，未来基本不会再调用。
> 保留在此供历史回溯，但不属于日常工作流。

## 文件清单

### 金标演进（用于 gold/gold_triples_augmented.csv 的生成）

| 脚本 | 用途 | 产物 |
|------|------|------|
| `align_gold_to_pred.py` | 把 gold 中 head/tail 与 pred 子串规约对齐 | `gold/gold_triples_aligned.csv` |
| `augment_gold_with_instance_of.py` | 把 pred 中合理的 `instance_of` 反向补入 gold | `gold/gold_triples_augmented.csv`（主基线） |

### LLM NER 实验（已回滚）

| 脚本 | 用途 |
|------|------|
| `llm_ner_expand.py` | 用 LLM 全文扫描，找出未登录的领域实体 |
| `audit_llm_ner.py` | 对 LLM 提议的实体做规则审计与重打类型 |
| `merge_ner_terms.py` | 把审计后的实体并入 `data/entities_by_type.json` |
| `prune_long_ner.py` | 剔除"吞并"短实体的过长复合 NER 词 |
| `suggest_ner_terms.py` | 从 FN 推荐 NER 词典扩充 |

> 结论：LLM NER 扩充在评估中表现不如预期（长复合词吞并短实体造成 P 降），整体回滚。
> 工具保留供未来在新语料上重试。

### 错误分析

| 脚本 | 用途 |
|------|------|
| `error_analysis.py` | 对 FP/FN 做关系分布统计 + 混淆矩阵分析 |

## 如何重新启用

如果未来要用其中任何一个，直接 import：

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "archive"))
from align_gold_to_pred import main as align_main
```

或者把脚本移回 `tools/` 即可。
