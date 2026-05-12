"""根据第四章评估的错误分析，向 NER 词典推荐新增的实体（待人工审查）。

来源：
    - gold/gold_triples.csv 中所有出现的实体（head / tail）
    - 排除 data/entities_by_type.json 已收录的实体
    - 排除数值型实体（年份、带单位数值）

类型推荐策略（按优先级）：
    1) 直接利用 gold 里 `<X, instance_of, <类型>>` 的标签
       （gold 里 instance_of 是权威类型）
    2) 关键字启发式（参数/翼/方法/公司/人名…）
    3) 缺省 CONCEPT

输出：
    data/entities_to_add_ch4.json  待审查清单（按 type 分组）
    output/ner_suggestions_ch4.md  人类可读报告（带 gold 三元组证据）

之后用户审查 -> 合并到 data/entities_by_type.json 或 ans.py。
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# gold 里 instance_of 的中文 tail（粗类型名） → entities_by_type.json 的英文 type
_ZH_TO_ETYPE: dict[str, str] = {
    "飞行器": "AIRCRAFT",
    "机翼构型": "WING_CONFIGURATION",
    "设计参数": "PARAMETER",
    "气动概念": "AERODYNAMIC_CONCEPT",
    "结构部件": "STRUCTURAL_COMPONENT",
    "控制方法": "CONTROL_METHOD",
    "性能指标": "PERFORMANCE_METRIC",
    "组织机构": "ORGANIZATION",
    "人物": "PERSON",
    "技术": "TECHNOLOGY",
    "材料": "MATERIAL",
    "飞行阶段": "FLIGHT_PHASE",
    "公式": "EQUATION",
    "概念": "CONCEPT",
    "翼型": "AIRCRAFT",  # 翼型在现有 14 类中没有专门类别，归入 AIRCRAFT 大类
}


_NUMBER_LIKE_RE = re.compile(
    r"^\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?"
    r"\s*(?:%|°C?|度|km/h|m/s|km|kg|kw|kW|t|g|m|cm|mm|s|ms|kn|N·m|Nm|N|Mach|马赫|"
    r"公里|米|秒|克|吨|公斤|赫兹|Hz|分钟|小时|h|次|步|倍|万|气|时)?$"
)
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}(?:年|s|年代)?$")


def is_numberish(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if _YEAR_RE.match(s):
        return True
    if _NUMBER_LIKE_RE.match(s):
        return True
    return False


# 数值/单位结尾扩展（更宽松，"10度每秒/3200N·m/0.005秒/7000N·m"等）
_VALUE_TAIL_RE = re.compile(
    r"^.*\d+(?:\.\d+)?\s*"
    r"(?:%|度|度/秒|度每秒|km/h|m/s|km|kg|kw|kW|t|g|m|cm|mm|s|ms|秒|N·m|Nm|N|万|倍|气|时|°C?|赫兹|Hz|公里|米|克|吨|公斤)$"
)


def heuristic_etype(name: str) -> str:
    """启发式：根据实体名关键字推断 type。"""
    n = name
    # 0) 完全是"数字+单位"短语 → 视为数值，不入词典（返回特殊标记）
    if _VALUE_TAIL_RE.match(n):
        return "_DROP_"  # 由上层过滤掉

    # 1) 人名（大写英文 + 没有空格的西文姓氏）
    if re.match(r"^[A-Z][a-zA-Z]+$", n) and len(n) <= 14:
        return "PERSON"
    # 2) 翼型代号 NACA0012 / NACA2412 / Boeing106
    if re.match(r"^[A-Z]+\d+[A-Z]?$", n):
        return "AIRCRAFT"
    # 3) 机翼布局（要在 STRUCTURAL 之前匹配）
    if any(k in n for k in ("布局", "构型", "三维双翼", "二维双翼")):
        return "WING_CONFIGURATION"
    # 4) 公司 / 机构（先排除非机构关键字 "布局/构型" 已上面捕获）
    if any(k in n for k in ("公司", "集团", "局", "院", "中心", "研究所", "实验室", "NASA", "DARPA", "波音", "空客", "通用原子", "Aeronautics", "Boeing", "Airbus")):
        return "ORGANIZATION"
    # 5) 位置描述短语 → 视为概念（不是结构部件）
    if any(n.endswith(suf) for suf in ("前方", "后方", "上方", "正上方", "斜前方", "斜后方", "两端", "顶部", "底部", "侧面", "侧前方", "侧后方", "之间", "中心")):
        return "CONCEPT"
    # 6) 翼型代号兜底
    if "翼型" in n:
        return "AIRCRAFT"
    # 7) 结构部件
    if any(k in n for k in ("翼肋", "翼梁", "蒙皮", "桁架", "翼板", "副翼", "翼面", "翼根", "翼尖", "支柱", "张线", "U型槽", "纵墙", "长桁", "主梁", "铰链", "嵌套网格", "背景网格", "对称面", "端面", "网格")):
        return "STRUCTURAL_COMPONENT"
    # 8) 设计参数（值/性质名 — 带"长度/重量/最大/最小/系数/角度/速度/总时间/比/数值"）
    if any(k in n for k in (
        "弦长", "翼展", "间距", "上反角", "攻角", "安装角", "面积", "弯度", "梢根比",
        "展弦比", "翼展比", "雷诺数", "马赫数", "动压", "升力系数", "阻力系数",
        "升力线斜率", "因子", "压力", "时间步长", "重量", "长度", "扭矩", "弯矩",
        "速度", "总时间", "迭代次数", "功率", "角度", "公转角度", "最大", "最小",
    )):
        return "PARAMETER"
    # 9) 气动概念
    if any(k in n for k in (
        "升力", "阻力", "俯仰", "偏航", "滚转", "失速", "诱导阻力", "干扰", "下洗",
        "上洗", "涡", "翼尖涡", "气流", "流动分离", "压差", "压区", "高压区", "低压区",
        "气动",
    )):
        return "AERODYNAMIC_CONCEPT"
    # 10) 控制方法（"控制 + 不是名词"）
    if any(k in n for k in ("控制", "反馈", "PID", "滑模", "鲁棒", "自适应控制", "动力学建模")):
        return "CONTROL_METHOD"
    # 11) 性能指标
    if any(k in n for k in ("航程", "续航", "巡航速度", "升限", "起降距离", "载重", "推重比", "燃油效率")):
        return "PERFORMANCE_METRIC"
    # 12) 技术 / 方法 / 模型
    if any(k in n for k in ("方法", "技术", "算法", "模型", "理论", "计算", "仿真", "网格法", "RANS", "URANS", "DES", "LES", "CFD", "FEM")):
        return "TECHNOLOGY"
    # 13) 概念兜底
    return "CONCEPT"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    # ===== 加载现有词典 =====
    with open(os.path.join(ROOT, "data", "entities_by_type.json"), encoding="utf-8-sig") as f:
        existing: dict[str, list[str]] = json.load(f)
    existing_terms: set[str] = set()
    for v in existing.values():
        for w in v:
            existing_terms.add((w or "").strip())

    # ===== 加载 gold 三元组 =====
    gold_rows: list[tuple[str, str, str]] = []
    with open(os.path.join(ROOT, "gold", "gold_triples.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            h = (r.get("head") or "").strip()
            rel = (r.get("relation") or "").strip()
            t = (r.get("tail") or "").strip()
            if h and rel and t:
                gold_rows.append((h, rel, t))

    # ===== gold 内部的 instance_of 权威类型表 =====
    instance_of_map: dict[str, str] = {}
    for h, rel, t in gold_rows:
        if rel == "instance_of":
            instance_of_map[h] = t

    # ===== 收集 gold 所有实体 → 减去 existing → 减去数值型 =====
    gold_ents: set[str] = set()
    for h, _, t in gold_rows:
        gold_ents.add(h)
        gold_ents.add(t)

    candidates: list[str] = []
    for e in sorted(gold_ents):
        if e in existing_terms:
            continue
        if is_numberish(e):
            continue
        # 跳过粗类型词本身（如 "设计参数" 是 type 名，不该作实体）
        if e in _ZH_TO_ETYPE:
            continue
        candidates.append(e)

    # ===== 为每个候选推荐 type + 收集证据 =====
    suggestions: dict[str, list[dict]] = defaultdict(list)
    dropped: list[dict] = []
    for ent in candidates:
        # 优先：gold instance_of 标签
        if ent in instance_of_map:
            zh = instance_of_map[ent]
            etype = _ZH_TO_ETYPE.get(zh, heuristic_etype(ent))
            source = f"gold instance_of=({zh})"
        else:
            etype = heuristic_etype(ent)
            source = "heuristic"
        # 收集该实体作为 head/tail 出现的 gold 三元组（最多 3 条作为证据）
        evidence: list[str] = []
        for h, r, t in gold_rows:
            if h == ent or t == ent:
                evidence.append(f"{h} —[{r}]→ {t}")
                if len(evidence) >= 3:
                    break
        if etype == "_DROP_":
            dropped.append({"term": ent, "reason": "纯数值/单位短语", "evidence": evidence})
            continue
        suggestions[etype].append({
            "term": ent,
            "source": source,
            "evidence": evidence,
        })

    # ===== 写 JSON =====
    out_json = {k: sorted(v, key=lambda x: x["term"]) for k, v in sorted(suggestions.items())}
    if dropped:
        out_json["_DROPPED_NUMERIC_"] = sorted(dropped, key=lambda x: x["term"])
    out_path_json = os.path.join(ROOT, "data", "entities_to_add_ch4.json")
    with open(out_path_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # ===== 写 markdown 报告 =====
    md: list[str] = []
    md.append("# 第四章 · NER 词典扩充建议\n")
    md.append(f"- 候选总数: **{sum(len(v) for v in suggestions.values())}**")
    md.append(f"- 已收录实体数: **{len(existing_terms)}**（来自 ans.EntityLibrary 导出）")
    md.append("")

    md.append("## 按 type 分布\n")
    md.append("| type | 新增数 | 现有数 |")
    md.append("|------|------:|------:|")
    for etype in sorted(suggestions):
        old = len(existing.get(etype, []))
        md.append(f"| {etype} | {len(suggestions[etype])} | {old} |")
    md.append(f"| _DROPPED_NUMERIC_ | {len(dropped)} | (数值/单位短语，不入词典) |")
    md.append("")

    for etype in sorted(suggestions):
        md.append(f"\n## {etype}  （新增 {len(suggestions[etype])} 个）\n")
        for item in suggestions[etype]:
            ev = "  ".join([f"`{x}`" for x in item["evidence"]]) or "(无 gold 直接证据)"
            md.append(f"- **{item['term']}**  <sub>{item['source']}</sub>")
            md.append(f"  - {ev}")

    md.append("\n---\n")
    md.append("## 审查方式\n")
    md.append("1. 浏览 `data/entities_to_add_ch4.json`，删除错分/不需要的条目")
    md.append("2. 调整 type（如把误分到 CONCEPT 的术语手动改到 PARAMETER）")
    md.append("3. 确认后由我合并到 `data/entities_by_type.json` 并重跑评估对比涨幅")

    out_path_md = os.path.join(ROOT, "output", "ner_suggestions_ch4.md")
    os.makedirs(os.path.dirname(out_path_md), exist_ok=True)
    with open(out_path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[suggest_ner] 候选总数: {sum(len(v) for v in suggestions.values())}")
    print(f"[suggest_ner] 已写 JSON:     {out_path_json}")
    print(f"[suggest_ner] 已写 Markdown: {out_path_md}")
    print("[suggest_ner] 按 type 分布:")
    for etype in sorted(suggestions):
        print(f"  {etype:24s} +{len(suggestions[etype])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
