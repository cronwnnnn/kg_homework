"""自动抽取结果 vs 人工金标 评估器（多口径 + 错误分析）。

四个评估口径：
    1) **严格 F1 (L1)**: (head, relation, tail) 全匹配；
    2) **宽松 F1 (L2)**: 仅 (head, tail) 匹配（忽略关系名）；
    3) **Partial F1 (L3)**: 关系一致 + head/tail 双向子串匹配（min_len=2）；
    4) **实体级 F1**: 端点实体集合 P/R/F1。

默认输入：
    --gold gold/gold_triples_augmented.csv     # 第四章主基线金标（745 条）

按章节过滤（仅评估某一章的指标）：
    --chapter "第4章"            # 子串匹配 pred CSV 的 chapter 列
    --include-global             # 同时纳入 instance_of 等全局类型分类
用法：
    uv run python evaluate_kg.py
    uv run python evaluate_kg.py --gold gold/gold_triples.csv --chapter "第4章" \
        --report output/eval_report_ch4.txt --out-dir output/ch4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Iterable


# 中文 / 不规范关系名 → 统一英文关系名（用于人工标注混入中文/英文混写的场景）
_REL_NORMALIZE_MAP: dict[str, str] = {
    "组成": "has_part",
    "包含": "has_part",
    "包括": "has_part",
    "产生": "generates",
    "造成": "causes",
    "导致": "causes",
    "影响": "affects",
    "减少": "reduces",
    "降低": "reduces",
    "增加": "increases",
    "提高": "improves",
    "提升": "improves",
    "改善": "improves",
    "属于": "is_a",
    "是一种": "instance_of",
    "位于": "located_at",
    "用于": "used_for",
    "等于": "equivalent_to",
    "定义为": "defined_as",
    "has": "has_part",   # 不规范，统一为 has_part
}


def normalize_relation(rel: str) -> str:
    rel = (rel or "").strip()
    return _REL_NORMALIZE_MAP.get(rel, rel)


# 全局型分类关系（即使 chapter 为空也属于"全图通用"，做章节评估时可选保留）
_GLOBAL_RELATIONS = frozenset({"instance_of", "is_a", "type_taxonomy"})


def load_aliases(path: str) -> dict[str, str]:
    """加载同义词表 (canonical → aliases list) → 反向字典 (alias → canonical)。

    JSON 格式：
        {
          "canonical_name": ["alias1", "alias2", ...],
          ...
        }
    以下划线开头的键（如 _comment / _format）会被忽略。
    返回的字典中，canonical 自身也会映射到自己，方便直接 .get(x, x)。
    """
    out: dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[eval] aliases 加载失败: {exc}", file=sys.stderr)
        return out
    if not isinstance(raw, dict):
        return out
    for canonical, aliases in raw.items():
        if canonical.startswith("_"):
            continue
        canonical = (canonical or "").strip()
        if not canonical:
            continue
        out.setdefault(canonical, canonical)
        if isinstance(aliases, list):
            for a in aliases:
                a = (a or "").strip()
                if not a or a == canonical:
                    continue
                # 后注册的不覆盖：一个 alias 只属于第一个声明它的 canonical
                out.setdefault(a, canonical)
    return out


def normalize_entity(name: str, aliases: dict[str, str]) -> str:
    if not aliases:
        return name
    return aliases.get((name or "").strip(), name)


def _read_triples_csv(
    path: str,
    chapter_keyword: str | None = None,
    normalize_rel: bool = True,
    include_global: bool = False,
    aliases: dict[str, str] | None = None,
) -> set[tuple[str, str, str]]:
    """读取三元组 CSV。

    chapter_keyword: 若给定且 CSV 含 ``chapter`` 列，则仅保留 chapter 包含该子串的行。
    normalize_rel:   是否把中文/不规范关系名映射为统一英文关系名。
    include_global:  在 chapter 过滤模式下，是否额外保留 chapter 为空且关系属于
                     instance_of/is_a/type_taxonomy 的"全局类型分类"行。
    """
    out: set[tuple[str, str, str]] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
        has_chapter_col = ("chapter" in fieldnames) or ("章节" in fieldnames)

        def pick(row: dict[str, str], *keys: str) -> str:
            for k in keys:
                if k in row and row[k]:
                    return (row[k] or "").strip()
            return ""

        for row in reader:
            if chapter_keyword and has_chapter_col:
                ch = (row.get("chapter") or row.get("章节") or "").strip()
                rel_raw = (row.get("relation") or row.get("关系") or "").strip()
                rel_norm = normalize_relation(rel_raw) if normalize_rel else rel_raw
                if ch:
                    if chapter_keyword not in ch:
                        continue
                else:
                    if not (include_global and rel_norm in _GLOBAL_RELATIONS):
                        continue
            # 若 CSV 没有 chapter 列（如 gold_triples.csv），不做章节过滤
            h = pick(row, "head", "头实体", "source", "h")
            r = pick(row, "relation", "关系", "type", "r")
            t = pick(row, "tail", "尾实体", "target", "t")
            if normalize_rel:
                r = normalize_relation(r)
            if aliases:
                h = normalize_entity(h, aliases)
                t = normalize_entity(t, aliases)
            if h and r and t and h != t:
                out.add((h, r, t))
    return out


def _read_entities_txt(path: str) -> set[str]:
    s: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                s.add(w)
    return s


def _entities_from_triples(triples: Iterable[tuple[str, str, str]]) -> set[str]:
    bag: set[str] = set()
    for h, _, t in triples:
        bag.add(h)
        bag.add(t)
    return bag


def prf1(pred: set, gold: set) -> tuple[float, float, float, int]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0, 0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp


def _partial_token_match(a: str, b: str, min_len: int = 2) -> bool:
    """token-level partial: 完全相等 或 一方是另一方的子串（双方都需 >= min_len 字）。

    例：'连杆' 与 '连杆最大扭矩' 互为子串 → True
        'A' 与 'AB' 因 'A' 过短 → False（防止"机"匹配所有含"机"的实体）
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    if len(a) < min_len or len(b) < min_len:
        return False
    return a == b or (a in b) or (b in a)


def partial_prf1(
    pred: set[tuple[str, str, str]],
    gold: set[tuple[str, str, str]],
    min_len: int = 2,
) -> tuple[float, float, float, int]:
    """L3 Partial F1：关系一致 + head/tail 子串双向匹配。

    匹配规则：
      - relation 必须完全相等（不放宽关系）；
      - head 与 gold_head 互为子串（任一方为另一方子串即可）；
      - tail 与 gold_tail 互为子串；
      - 每条 gold 三元组最多被一条 pred 匹配（greedy + matched 集合避免一对多虚高）。

    适用场景：head 命名颗粒度不同导致严格 F1 偏低时，做"语义级"放宽。
    报告里与严格 F1、宽松 F1 并列展示，不替代它们。
    """
    if not pred and not gold:
        return 1.0, 1.0, 1.0, 0
    gold_list = list(gold)
    matched_gold: set[int] = set()
    tp = 0
    for p_head, p_rel, p_tail in pred:
        for gi, (g_head, g_rel, g_tail) in enumerate(gold_list):
            if gi in matched_gold:
                continue
            if p_rel != g_rel:
                continue
            if not _partial_token_match(p_head, g_head, min_len):
                continue
            if not _partial_token_match(p_tail, g_tail, min_len):
                continue
            tp += 1
            matched_gold.add(gi)
            break
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="知识图谱抽取 vs 人工金标 多口径评估")
    p.add_argument("--pred", default="output/triples_with_meta.csv", help="预测三元组 CSV")
    p.add_argument(
        "--gold",
        default="gold/gold_triples_augmented.csv",
        help="人工金标三元组 CSV（默认为第四章主基线 augmented 版）",
    )
    p.add_argument(
        "--gold-entities",
        default="gold/gold_entities.csv",
        help="实体金标列表（可选）；不存在则从金标三元组端点推导",
    )
    p.add_argument(
        "--top-relations",
        type=int,
        default=15,
        help="关系级 P/R/F1 仅展示出现频次 Top-N 的关系（默认 15）",
    )
    p.add_argument(
        "--chapter",
        default="",
        help="只评估 chapter 字段包含此子串的预测三元组（如 '第4章'）；gold 若无 chapter 列则不过滤",
    )
    p.add_argument(
        "--no-normalize-rel",
        action="store_true",
        help="禁用中文关系名归一化（默认会把 组成/产生/造成/影响/has 等映射为 has_part/generates/...）",
    )
    p.add_argument(
        "--include-global",
        action="store_true",
        help="章节评估模式下，额外保留 chapter 为空且关系是 instance_of/is_a/type_taxonomy 的全局类型分类行",
    )
    p.add_argument(
        "--aliases-file",
        default="",
        help="同义词表 JSON（canonical→aliases）。比对前把 head/tail 中的别名规约为 canonical",
    )
    p.add_argument(
        "--exclude-relations",
        default="discussed_in,co_occurs_with",
        help="评估时从 pred 和 gold 中剔除这些关系（逗号分隔）。默认排除 pred 独有的元数据关系 discussed_in / co_occurs_with",
    )
    p.add_argument("--report", default="output/eval_report.txt", help="评估报告路径")
    p.add_argument("--out-dir", default="output", help="错误分析 CSV 输出目录")
    return p.parse_args()


def _entity_pairs(triples: set[tuple[str, str, str]]) -> set[tuple[str, str]]:
    return {(h, t) for h, _r, t in triples}


def _relation_breakdown(
    pred: set[tuple[str, str, str]],
    gold: set[tuple[str, str, str]],
    top_n: int,
) -> list[tuple[str, int, int, int, float, float, float]]:
    """按关系计算 P/R/F1（仅看金标中出现的关系，按金标频次取 Top-N）。"""
    gold_rel_counter = Counter(r for _, r, _ in gold)
    rels = [r for r, _ in gold_rel_counter.most_common(top_n)]
    out: list[tuple[str, int, int, int, float, float, float]] = []
    for r in rels:
        pred_r = {t for t in pred if t[1] == r}
        gold_r = {t for t in gold if t[1] == r}
        p, rec, f1, tp = prf1(pred_r, gold_r)
        out.append((r, len(pred_r), len(gold_r), tp, p, rec, f1))
    return out


def _confusion_top(
    pred: set[tuple[str, str, str]],
    gold: set[tuple[str, str, str]],
    top_n: int = 12,
) -> list[tuple[str, str, int]]:
    """实体对相同但关系不同的混淆 Top-N：(gold_rel, pred_rel, count)。"""
    gold_pairs: dict[tuple[str, str], str] = {(h, t): r for h, r, t in gold}
    cnt: Counter[tuple[str, str]] = Counter()
    for h, r, t in pred:
        gr = gold_pairs.get((h, t))
        if gr is not None and gr != r:
            cnt[(gr, r)] += 1
    return [(gr, pr, n) for (gr, pr), n in cnt.most_common(top_n)]


def _write_csv(path: str, rows: Iterable[tuple[str, str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "relation", "tail"])
        for h, r, t in sorted(rows):
            w.writerow([h, r, t])


def main() -> int:
    # Windows PowerShell 默认 cp936 控制台无法打印某些 Unicode 字符；
    # 这里把 stdout/stderr 切到 utf-8（errors=replace 兜底），仅影响打印，文件仍是 utf-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    args = parse_args()
    if not os.path.exists(args.pred):
        print(f"[eval] 找不到预测文件: {args.pred}", file=sys.stderr)
        return 1
    if not os.path.exists(args.gold):
        print(f"[eval] 找不到金标文件: {args.gold}", file=sys.stderr)
        print("[eval] 请用 --gold 显式指定金标 CSV 路径。", file=sys.stderr)
        return 1

    normalize_rel = not args.no_normalize_rel
    chapter_kw = (args.chapter or "").strip() or None

    aliases = load_aliases(args.aliases_file) if args.aliases_file else {}
    n_alias_pairs = sum(1 for k, v in aliases.items() if k != v) if aliases else 0

    excluded_rels: set[str] = set()
    if args.exclude_relations:
        excluded_rels = {x.strip() for x in args.exclude_relations.split(",") if x.strip()}

    def _filter_excluded(rows: set[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
        if not excluded_rels:
            return rows
        return {(h, r, t) for (h, r, t) in rows if r not in excluded_rels}

    pred_full = _filter_excluded(_read_triples_csv(
        args.pred,
        chapter_keyword=chapter_kw,
        normalize_rel=normalize_rel,
        include_global=args.include_global,
        aliases=aliases,
    ))
    gold_full = _filter_excluded(_read_triples_csv(
        args.gold,
        chapter_keyword=chapter_kw,
        normalize_rel=normalize_rel,
        include_global=args.include_global,
        aliases=aliases,
    ))

    pred = pred_full
    gold = gold_full

    # === 严格 / 宽松 / Partial / 实体对 四口径 ===
    p_strict, r_strict, f_strict, tp_strict = prf1(pred, gold)
    pred_pairs = _entity_pairs(pred)
    gold_pairs = _entity_pairs(gold)
    p_pair, r_pair, f_pair, tp_pair = prf1(pred_pairs, gold_pairs)
    p_part, r_part, f_part, tp_part = partial_prf1(pred, gold)

    # === 实体端点 ===
    # 当用户做章节过滤时，全文实体清单不适合作为参照，
    # 强制从过滤后的 gold 三元组端点推导，避免分母被拉高。
    if chapter_kw or not os.path.isfile(args.gold_entities):
        gold_entities = _entities_from_triples(gold)
        gold_ent_source = "(从 gold 三元组端点推导)"
    else:
        gold_entities = _read_entities_txt(args.gold_entities)
        gold_ent_source = f"(读自 {args.gold_entities})"
    pred_entities = _entities_from_triples(pred)
    p_e, r_e, f_e, tp_e = prf1(pred_entities, gold_entities)

    # === 关系级 P/R/F1 ===
    rel_rows = _relation_breakdown(pred, gold, args.top_relations)

    # === TP / FP / FN ===
    tp_set = pred & gold
    fp_set = pred - gold
    fn_set = gold - pred

    macro_f1 = 0.0
    if rel_rows:
        macro_f1 = sum(r[6] for r in rel_rows) / len(rel_rows)

    confusion = _confusion_top(pred, gold)

    # === 报告 ===
    lines: list[str] = []
    lines.append("==== 知识图谱抽取评估 ====")
    lines.append(f"预测文件: {args.pred}")
    lines.append(f"金标文件: {args.gold}")
    if chapter_kw:
        extra = "；含全局类型分类 (chapter 空 + instance_of/is_a/type_taxonomy)" if args.include_global else ""
        lines.append(f"章节过滤: chapter 包含 {chapter_kw!r} 的预测三元组{extra}")
    if normalize_rel:
        lines.append("关系归一: 已启用 (中文/has → 统一英文关系名；用 --no-normalize-rel 关闭)")
    if aliases:
        lines.append(f"实体同义规约: 已启用 {args.aliases_file}  ({n_alias_pairs} 个别名 → canonical)")
    if excluded_rels:
        lines.append(f"关系剔除: 评估时排除 {sorted(excluded_rels)}")
    lines.append("")

    lines.append("---- 三元组 / 实体对 / 实体 四口径 ----")
    lines.append(f"严格 F1   (L1 三元组全相等):       P={p_strict:.4f}  R={r_strict:.4f}  F1={f_strict:.4f}  "
                 f"TP={tp_strict}  pred={len(pred)}  gold={len(gold)}")
    lines.append(f"宽松 F1   (L2 仅 head,tail 一致):  P={p_pair:.4f}  R={r_pair:.4f}  F1={f_pair:.4f}  "
                 f"TP={tp_pair}  pred_pairs={len(pred_pairs)}  gold_pairs={len(gold_pairs)}")
    lines.append(f"Partial F1(L3 head/tail 子串匹配): P={p_part:.4f}  R={r_part:.4f}  F1={f_part:.4f}  "
                 f"TP={tp_part}  (关系一致 + head/tail 双向子串，min_len=2)")
    lines.append(f"实体级 F1 (端点集合):             P={p_e:.4f}  R={r_e:.4f}  F1={f_e:.4f}  "
                 f"TP={tp_e}  pred_ent={len(pred_entities)}  gold_ent={len(gold_entities)} {gold_ent_source}")
    lines.append("")

    if rel_rows:
        lines.append(f"---- 关系级 P/R/F1 (金标频次 Top-{args.top_relations}) ----")
        lines.append(f"  {'关系':<22} {'pred':>6} {'gold':>6} {'TP':>5}   P     R     F1")
        for rel, np_, ng, tp, pp, rr, ff in rel_rows:
            lines.append(f"  {rel:<22} {np_:>6} {ng:>6} {tp:>5}  {pp:.3f} {rr:.3f} {ff:.3f}")
        lines.append(f"  宏平均 F1 (Top-{len(rel_rows)} 关系): {macro_f1:.4f}")
        lines.append("")

    if confusion:
        lines.append("---- 实体对相同但关系不同 (Top-12 混淆) ----")
        lines.append(f"  {'gold_rel':<18} -> {'pred_rel':<18}  cnt")
        for gr, pr, n in confusion:
            lines.append(f"  {gr:<18} -> {pr:<18}  {n}")
        lines.append("")

    lines.append(f"---- 错误分析样本 ----")
    lines.append(f"TP (正确): {len(tp_set)} 条")
    lines.append(f"FP (误抽): {len(fp_set)} 条")
    lines.append(f"FN (漏抽): {len(fn_set)} 条")

    if chapter_kw:
        lines.append("")
        lines.append("---- 章节评估注意事项 ----")
        lines.append(f"- pred 仅保留 chapter 含 {chapter_kw!r} 的三元组" + (
            "（含 chapter 空但关系是全局类型分类的行）。"
            if args.include_global
            else "；chapter 为空的类型分类 (instance_of 等) 默认被过滤，"
        ))
        if not args.include_global:
            lines.append("  这会让 instance_of 类 FN 偏高。如需把全局类型分类纳入评估，请加 --include-global。")
        lines.append("- 实体级 F1 已强制从过滤后的 gold 三元组端点推导参照集合。")

    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[eval] 已写: {args.report}")

    _write_csv(os.path.join(args.out_dir, "eval_tp.csv"), tp_set)
    _write_csv(os.path.join(args.out_dir, "eval_fp.csv"), fp_set)
    _write_csv(os.path.join(args.out_dir, "eval_fn.csv"), fn_set)
    print(f"[eval] 已写: {os.path.join(args.out_dir, 'eval_tp.csv')} / eval_fp.csv / eval_fn.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
