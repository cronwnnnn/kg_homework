"""第四章评估错误分析报告。

输入：
    --pred   output/triples_with_meta.csv
    --gold   gold/gold_triples.csv
    --chapter 第4章
    --include-global  (默认开启，与最终评估一致)

输出：
    output/error_analysis_ch4.md   人类可读的错误分类报告

报告分析维度：
    - FN：漏抽，按"实体是否被识别"分类
    - FP：误抽，按 head 是否为数值/年份、是否在 gold 实体中分类
    - 关系级 FN/FP 分布
    - 列出最常见的"未识别 gold 实体"（指导 NER 词典扩充）
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluate_kg  # noqa: E402


# 判断"纯数值/年份/百分比/物理量"作为主语
_NUMBER_LIKE_RE = re.compile(
    r"^"
    r"\d+(?:\.\d+)?"               # 数字
    r"(?:\s*-\s*\d+(?:\.\d+)?)?"   # 范围 5-10
    r"\s*(?:%|°C?|度|km/h|m/s|km|kg|kw|kW|t|g|m|cm|mm|s|ms|kn|N·m|Nm|N|Mach|马赫|公里|米|秒|克|吨|公斤|赫兹|Hz|分钟|小时|h|次|步|倍|万|气|时)?"
    r"$"
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


def read_csv_triples(path: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    (r.get("head") or "").strip(),
                    (r.get("relation") or "").strip(),
                    (r.get("tail") or "").strip(),
                )
            )
    return rows


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="output/triples_with_meta.csv")
    ap.add_argument("--gold", default="gold/gold_triples.csv")
    ap.add_argument("--chapter", default="第4章")
    ap.add_argument("--no-include-global", action="store_true")
    ap.add_argument("--report", default="output/error_analysis_ch4.md")
    ap.add_argument("--fp-dir", default="output/ch4_g/eval_fp.csv")
    ap.add_argument("--fn-dir", default="output/ch4_g/eval_fn.csv")
    ap.add_argument("--tp-dir", default="output/ch4_g/eval_tp.csv")
    args = ap.parse_args()

    include_global = not args.no_include_global

    pred = evaluate_kg._read_triples_csv(
        args.pred,
        chapter_keyword=args.chapter,
        normalize_rel=True,
        include_global=include_global,
    )
    gold = evaluate_kg._read_triples_csv(
        args.gold,
        chapter_keyword=args.chapter,
        normalize_rel=True,
        include_global=include_global,
    )
    pred_entities = {h for h, _, _ in pred} | {t for _, _, t in pred}
    gold_entities = {h for h, _, _ in gold} | {t for _, _, t in gold}

    tp = read_csv_triples(args.tp_dir)
    fp = read_csv_triples(args.fp_dir)
    fn = read_csv_triples(args.fn_dir)

    # ===== FN 分类 =====
    fn_cat = Counter()
    fn_by_rel = Counter()
    fn_samples_by_cat: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for h, r, t in fn:
        fn_by_rel[r] += 1
        h_in = h in pred_entities
        t_in = t in pred_entities
        if h_in and t_in:
            cat = "A. 两端实体都识别了 → 关系/规则缺失"
        elif h_in and not t_in:
            cat = "B. head 识别 / tail 漏识"
        elif (not h_in) and t_in:
            cat = "C. tail 识别 / head 漏识"
        else:
            cat = "D. 两端实体都没识别"
        fn_cat[cat] += 1
        if len(fn_samples_by_cat[cat]) < 6:
            fn_samples_by_cat[cat].append((h, r, t))

    # ===== FP 分类 =====
    fp_cat = Counter()
    fp_by_rel = Counter()
    fp_samples_by_cat: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for h, r, t in fp:
        fp_by_rel[r] += 1
        h_num = is_numberish(h)
        t_num = is_numberish(t)
        h_in_gold = h in gold_entities
        t_in_gold = t in gold_entities
        if h_num:
            cat = "A. head 是数值/年份 → 抽取边界错"
        elif t_num and r == "has_part":
            cat = "B. has_part 的 tail 是数值 → 应该是 has_value"
        elif h_in_gold and t_in_gold:
            cat = "C. 两端都是 gold 实体但关系不对"
        elif h_in_gold or t_in_gold:
            cat = "D. 一端是 gold 实体 / 一端误抓"
        else:
            cat = "E. 两端都不在 gold → 完全误抓"
        fp_cat[cat] += 1
        if len(fp_samples_by_cat[cat]) < 6:
            fp_samples_by_cat[cat].append((h, r, t))

    # ===== gold 中常见的未识别实体（指导 NER 词典扩充）=====
    gold_only_ents = gold_entities - pred_entities
    gold_only_num = [e for e in gold_only_ents if is_numberish(e)]
    gold_only_nonnum = sorted([e for e in gold_only_ents if not is_numberish(e)])

    # ===== 写报告 =====
    lines: list[str] = []
    lines.append("# 第四章评估 · 错误分析报告\n")
    lines.append(f"- pred: `{args.pred}`")
    lines.append(f"- gold: `{args.gold}`")
    lines.append(f"- chapter: `{args.chapter}`  include_global={include_global}")
    lines.append("")
    lines.append("## 总览\n")
    lines.append(f"- pred 三元组: **{len(pred)}**，pred 实体: **{len(pred_entities)}**")
    lines.append(f"- gold 三元组: **{len(gold)}**，gold 实体: **{len(gold_entities)}**")
    lines.append(f"- 实体重叠: **{len(pred_entities & gold_entities)}** "
                 f"(占 gold 的 {len(pred_entities & gold_entities)/max(len(gold_entities),1):.1%})")
    lines.append(f"- TP / FP / FN: **{len(tp)} / {len(fp)} / {len(fn)}**")
    lines.append("")

    # ===== FN =====
    lines.append("## FN（漏抽）共 {} 条\n".format(len(fn)))
    lines.append("### 按错因分类\n")
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|------|-----:|-----:|")
    for cat, n in fn_cat.most_common():
        lines.append(f"| {cat} | {n} | {n/len(fn):.1%} |")
    lines.append("")

    lines.append("### 各类别样例（最多 6 条）\n")
    for cat in fn_cat:
        lines.append(f"**{cat}**")
        lines.append("")
        for h, r, t in fn_samples_by_cat[cat]:
            lines.append(f"- `{h}` —[{r}]→ `{t}`")
        lines.append("")

    lines.append("### FN 按关系类型分布（Top-15）\n")
    lines.append("| relation | FN |")
    lines.append("|----------|---:|")
    for r, n in fn_by_rel.most_common(15):
        lines.append(f"| {r} | {n} |")
    lines.append("")

    # ===== FP =====
    lines.append("## FP（误抽）共 {} 条\n".format(len(fp)))
    lines.append("### 按错因分类\n")
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|------|-----:|-----:|")
    for cat, n in fp_cat.most_common():
        lines.append(f"| {cat} | {n} | {n/len(fp):.1%} |")
    lines.append("")

    lines.append("### 各类别样例（最多 6 条）\n")
    for cat in fp_cat:
        lines.append(f"**{cat}**")
        lines.append("")
        for h, r, t in fp_samples_by_cat[cat]:
            lines.append(f"- `{h}` —[{r}]→ `{t}`")
        lines.append("")

    lines.append("### FP 按关系类型分布（Top-15）\n")
    lines.append("| relation | FP |")
    lines.append("|----------|---:|")
    for r, n in fp_by_rel.most_common(15):
        lines.append(f"| {r} | {n} |")
    lines.append("")

    # ===== 未识别 gold 实体 =====
    lines.append("## gold 中未被 pred 识别的实体共 {} 个\n".format(len(gold_only_ents)))
    lines.append(f"- 其中数值型（建议忽略 / schema 调整）: **{len(gold_only_num)}** 个")
    lines.append(f"- 非数值型（建议扩充 NER 词典）: **{len(gold_only_nonnum)}** 个\n")
    lines.append("### 非数值型未识别 gold 实体（最多 60 个）\n")
    for e in gold_only_nonnum[:60]:
        lines.append(f"- `{e}`")
    if len(gold_only_nonnum) > 60:
        lines.append(f"- … 还有 {len(gold_only_nonnum) - 60} 个未列出")
    lines.append("")

    # ===== 结论 =====
    lines.append("## 结论 & 建议\n")
    fn_top_cat = fn_cat.most_common(1)[0]
    fp_top_cat = fp_cat.most_common(1)[0]
    lines.append(f"- FN 主因: **{fn_top_cat[0]}** ({fn_top_cat[1]/len(fn):.1%})")
    lines.append(f"- FP 主因: **{fp_top_cat[0]}** ({fp_top_cat[1]/len(fp):.1%})")
    lines.append("")
    lines.append("**改进策略推荐**：")
    lines.append("")
    lines.append("- 如果 FN 主因是 D（两端都没识别）→ 优先扩充 NER 词典")
    lines.append("- 如果 FN 主因是 A（实体识别了但关系缺失）→ 增加关系触发词 / 修关系抽取规则")
    lines.append("- 如果 FP 主因是 A（head 是数值/年份）→ 在抽取后处理过滤纯数值/年份主语")
    lines.append("- 如果 FP 主因是 C（关系错）→ 修关系方向 / 模板")
    lines.append("- gold 是否需要改？仅以下情况合法：")
    lines.append("  1. 发现事实错误（与论文原文矛盾）")
    lines.append("  2. 关系命名不规范（已通过归一化自动处理）")
    lines.append("  3. 想明确 schema 范围（如不抽事件状态短语）→ 这是设计决策，不是为涨分")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[error_analysis] 已写: {args.report}")
    print(f"[error_analysis] FN 分布: {dict(fn_cat)}")
    print(f"[error_analysis] FP 分布: {dict(fp_cat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
