"""审查 LLM NER 候选，输出过滤后的可合并清单与审查报告。

输入：data/entities_to_add_llm.json
输出：
  data/entities_to_add_llm_audited.json   过滤+改类型后的清单（格式与 merge_ner_terms.py 兼容）
  output/llm_ner_audit_report.md          人工可读的审查说明
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DROP: dict[str, str] = {
    "大展弦比飞行器": "与 CONCEPT 里同名项重复（且形容词+名词不适合做 AIRCRAFT）",
    "U-2高空侦察机": "U-2 已在 AIRCRAFT 词典，啰嗦版本不需要",
    "Aquila太阳能无人机": "Aquila 已在 AIRCRAFT 词典，啰嗦版本不需要",
    "大展弦比": "形容词修饰词，不是独立实体；展弦比本身已是已知 PARAMETER",
    "超大展弦比": "同上",
    "气动-结构重量最优的展弦比": "复合长名词，质量低，可由两个独立实体表达",
    "变形": "过于宽泛，不是可定位实体",
    "强度刚度": "并列复合词，应拆为强度/刚度",
    "初始总体参数": "宽泛短语，不是独立实体",
    "翼尖擦地": "事故现象不是气动概念，且评估收益小",
    "二维弦向气动特性": "研究内容描述，不是独立概念",
    "三维展向气动特性": "研究内容描述，不是独立概念",
    "厚翼型形状": "形状描述，不是独立部件名",
    "控制策略": "宽泛短语",
    "控制输入": "宽泛短语",
    "控制输出": "宽泛短语",
    "飞行控制方法": "宽泛短语",
    "双连杆布局": "与 STRUCTURAL_COMPONENT.双连杆结构 重复",
    "单连杆布局": "与 STRUCTURAL_COMPONENT.单连杆结构 重复",
    "飞行性能": "宽泛短语",
    "性能指标": "宽泛短语",
    "多学科总体设计工具": "与 多学科总体设计优化平台 同义",
    "数值计算方法": "宽泛短语，被计算流体力学覆盖",
    "多学科方法": "宽泛短语，被多学科总体设计方法覆盖",
    "舰载部署": "动作短语，不是实体节点",
    "动力学和控制设计": "并列复合词",
    "平面形状": "宽泛短语",
    "平面外形设计": "宽泛短语",
    "气动设计": "宽泛短语",
    "飞行控制设计": "宽泛短语",
    "设计要求": "宽泛短语",
    "大展弦比长航飞行器": "原文笔误（缺“时”），与大展弦比长航时飞行器重复",
}


RETYPE: dict[str, str] = {
    "NACA2412翼型": "STRUCTURAL_COMPONENT",
    "Boeing106翼型": "STRUCTURAL_COMPONENT",
    "NACA0008对称翼型": "STRUCTURAL_COMPONENT",
    "大型长航时飞机": "CONCEPT",
    "太阳能飞机": "CONCEPT",
    "翼型": "CONCEPT",
    "10自由度": "CONCEPT",
    "ACADO多学科总体设计平台": "TECHNOLOGY",
    "ACADO平台": "TECHNOLOGY",
    "空中折叠变体试验": "CONCEPT",
    "空中折叠和展开变体试验": "CONCEPT",
    "地面车载变体试验": "CONCEPT",
    "空中飞行变体试验": "CONCEPT",
    "基本飞行试验": "CONCEPT",
    "变体飞行试验": "CONCEPT",
    "车载变体试验": "CONCEPT",
    "鸭式布局": "WING_CONFIGURATION",
    "三翼面布局": "WING_CONFIGURATION",
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    src_path = os.path.join(ROOT, "data", "entities_to_add_llm.json")
    out_path = os.path.join(ROOT, "data", "entities_to_add_llm_audited.json")
    rpt_path = os.path.join(ROOT, "output", "llm_ner_audit_report.md")
    os.makedirs(os.path.dirname(rpt_path), exist_ok=True)

    with open(src_path, encoding="utf-8") as f:
        raw: dict[str, list[dict]] = json.load(f)

    audited: dict[str, list[dict]] = {}
    dropped: list[tuple[str, str, str]] = []  # (term, origin_type, reason)
    retyped: list[tuple[str, str, str]] = []  # (term, from_type, to_type)
    kept_total = 0

    for etype, items in raw.items():
        for it in items:
            term = (it.get("term") or "").strip()
            if not term:
                continue
            if term in DROP:
                dropped.append((term, etype, DROP[term]))
                continue
            target_type = RETYPE.get(term, etype)
            if target_type != etype:
                retyped.append((term, etype, target_type))
            audited.setdefault(target_type, []).append(it)
            kept_total += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audited, f, ensure_ascii=False, indent=2)

    lines: list[str] = []
    lines.append("# LLM NER 审查报告\n")
    lines.append(f"- 候选总数：{sum(len(v) for v in raw.values())}\n")
    lines.append(f"- 保留：**{kept_total}**\n")
    lines.append(f"- 删除：**{len(dropped)}**\n")
    lines.append(f"- 改类型：**{len(retyped)}**\n\n")

    lines.append("## 保留分布（合并后）\n")
    for et in sorted(audited):
        lines.append(f"- **{et}**: {len(audited[et])} 个\n")
    lines.append("\n")

    lines.append("## 删除（共 {} 条）\n".format(len(dropped)))
    for term, ot, reason in dropped:
        lines.append(f"- `{term}` (原 {ot})  ←  {reason}\n")
    lines.append("\n")

    lines.append("## 改类型（共 {} 条）\n".format(len(retyped)))
    for term, ft, tt in retyped:
        lines.append(f"- `{term}`  {ft} → **{tt}**\n")
    lines.append("\n")

    lines.append("## 保留清单（按目标类型）\n")
    for et in sorted(audited):
        lines.append(f"\n### {et}（{len(audited[et])}）\n")
        for it in audited[et]:
            term = it.get("term", "")
            ev = (it.get("evidence", "") or "").replace("\n", " ")[:120]
            lines.append(f"- `{term}` — {ev}\n")

    with open(rpt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[audit] 候选总数: {sum(len(v) for v in raw.values())}")
    print(f"[audit] 保留: {kept_total}  删除: {len(dropped)}  改类型: {len(retyped)}")
    print(f"[audit] 清单输出: {out_path}")
    print(f"[audit] 报告输出: {rpt_path}")
    print("\n下一步：审查 output/llm_ner_audit_report.md，确认无误后运行：")
    print("  uv run python tools/merge_ner_terms.py --add data/entities_to_add_llm_audited.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
