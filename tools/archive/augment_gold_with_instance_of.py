"""把 pred 里"客观正确"的 instance_of 三元组补进 gold。

理由：
    type_extractor 按 data/entities_by_type.json 自动把识别到的实体映射到 14 个固定
    类型（飞行器/结构部件/设计参数 等）。这些 (entity, instance_of, type) 在语义上
    都是对的，只是 gold 没把所有实体都标全。把合理的补进 gold 是公平的对齐。

过滤规则（剔除明显不该补的 head）：
    - head 为空 / 长度 < 2 / 长度 >= 12
    - head 纯数字 / 含 %/倍/单位/下划线
    - head 含 _ 后缀（如"公转角度_单翼状态"，应保留 gold 已有的版本）
    - head 与 tail 相同（"机翼 instance_of 机翼"无意义）
    - head 的 tail（类型）与 gold 已有该 head 的类型冲突（避免覆盖人工判断）

用法::
    uv run python tools/augment_gold_with_instance_of.py --dry-run
    uv run python tools/augment_gold_with_instance_of.py --apply
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RE_VALUE_LIKE = re.compile(r"%|倍|m/s|N·m|N/m|km/h|kg|m\^2|°|/")


def load_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def is_garbage_head(h: str, t: str, min_len: int = 2, max_len: int = 12) -> tuple[bool, str]:
    h = (h or "").strip()
    if not h:
        return True, "空"
    if len(h) < min_len:
        return True, f"<{min_len}字"
    if len(h) >= max_len:
        return True, f">={max_len}字（过长复合）"
    if h.isdigit():
        return True, "纯数字"
    if h and h[0].isdigit():
        return True, "数字开头"
    if _RE_VALUE_LIKE.search(h):
        return True, "含数值/单位/标点"
    if "_" in h:
        return True, "含下划线后缀（gold 已有更细版本）"
    if h == (t or "").strip():
        return True, "head==tail"
    return False, ""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--gold", default=os.path.join(ROOT, "gold", "gold_triples_aligned.csv"),
                   help="基础 gold 文件（默认对齐版）")
    p.add_argument("--pred", default=os.path.join(ROOT, "output", "triples_with_meta.csv"))
    p.add_argument("--out", default=os.path.join(ROOT, "gold", "gold_triples_augmented.csv"))
    p.add_argument("--report", default=os.path.join(ROOT, "output", "gold_augment_report.md"))
    p.add_argument("--min-head-len", type=int, default=2)
    p.add_argument("--max-head-len", type=int, default=12)
    p.add_argument("--chapter-only", action="store_true", default=True,
                   help="仅评估章节4 时，instance_of 来自全文且 chapter 为空，无须过滤章节")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    if not args.dry_run and not args.apply:
        args.dry_run = True
        print("[augment] 默认 --dry-run（如需写文件加 --apply）\n")

    gold_fields, gold_rows = load_csv(args.gold)
    _, pred_rows = load_csv(args.pred)

    # 收集 gold 已有的 (head, instance_of) → tail 映射
    gold_inst: dict[str, str] = {}
    gold_set: set[tuple[str, str, str]] = set()
    for r in gold_rows:
        h = (r.get("head") or "").strip()
        rel = (r.get("relation") or "").strip()
        t = (r.get("tail") or "").strip()
        gold_set.add((h, rel, t))
        if rel == "instance_of":
            gold_inst[h] = t

    # 收集 pred 里所有 instance_of（全文，因为 instance_of 是全局类型分类）
    pred_inst_rows: list[tuple[str, str, str]] = []
    for r in pred_rows:
        if (r.get("relation") or "").strip() != "instance_of":
            continue
        h = (r.get("head") or "").strip()
        t = (r.get("tail") or "").strip()
        if h and t:
            pred_inst_rows.append((h, "instance_of", t))

    # 去重
    pred_inst_set = set(pred_inst_rows)

    # 分类
    will_add: list[tuple[str, str]] = []
    skip_garbage: list[tuple[tuple[str, str, str], str]] = []
    skip_already: list[tuple[str, str, str]] = []
    skip_conflict: list[tuple[str, str, str, str]] = []  # (h, t, gold_t, reason)

    for h, rel, t in sorted(pred_inst_set):
        if (h, rel, t) in gold_set:
            skip_already.append((h, rel, t))
            continue
        bad, reason = is_garbage_head(h, t, args.min_head_len, args.max_head_len)
        if bad:
            skip_garbage.append(((h, rel, t), reason))
            continue
        if h in gold_inst and gold_inst[h] != t:
            skip_conflict.append((h, t, gold_inst[h], "gold 已为此 head 标了不同类型"))
            continue
        will_add.append((h, t))

    # 统计
    print("=" * 70)
    print(f"[augment] pred instance_of 唯一三元组: {len(pred_inst_set)} 条")
    print(f"[augment] 其中 gold 已有        : {len(skip_already)} 条")
    print(f"[augment] 跳过 head 是垃圾      : {len(skip_garbage)} 条")
    print(f"[augment] 跳过 head 类型冲突    : {len(skip_conflict)} 条")
    print(f"[augment] => 将补入 gold        : {len(will_add)} 条")
    print(f"[augment] gold 总条数 {len(gold_rows)} -> {len(gold_rows) + len(will_add)}")
    print("=" * 70)

    # 垃圾 reason 分布
    if skip_garbage:
        rc = Counter(r for _, r in skip_garbage)
        print("\n[augment] 跳过原因分布:")
        for reason, c in rc.most_common():
            print(f"  {reason:30s}  {c}")
        print("\n[augment] 跳过的垃圾样本（前 15 条）:")
        for (h, rel, t), reason in skip_garbage[:15]:
            print(f"  ({h}, {rel}, {t})  [{reason}]")

    if skip_conflict:
        print("\n[augment] 类型冲突样本（前 10 条）:")
        for h, t, gold_t, _ in skip_conflict[:10]:
            print(f"  pred: ({h}, instance_of, {t})  vs  gold: ({h}, instance_of, {gold_t})")

    print(f"\n[augment] 将补入 gold 的样本（前 20 条）:")
    for h, t in will_add[:20]:
        print(f"  ({h}, instance_of, {t})")

    if args.dry_run:
        print("\n[augment] dry-run 完成。如需写入加 --apply。")
        return 0

    # 备份
    backup = args.gold + ".before_augment.csv"
    if not os.path.exists(backup):
        shutil.copyfile(args.gold, backup)
        print(f"\n[augment] 已备份: {backup}")

    # 写入 augmented gold
    if "chapter" in gold_fields:
        new_fields = list(gold_fields)
    else:
        new_fields = list(gold_fields)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for r in gold_rows:
            writer.writerow(r)
        for h, t in will_add:
            row = {k: "" for k in new_fields}
            row["head"] = h
            row["relation"] = "instance_of"
            row["tail"] = t
            writer.writerow(row)
    print(f"\n[augment] 输出: {args.out}")

    # 报告
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Gold 补全 instance_of 报告\n\n")
        f.write(f"- 基础 gold: `{args.gold}` ({len(gold_rows)} 条)\n")
        f.write(f"- pred instance_of 唯一: {len(pred_inst_set)}\n")
        f.write(f"- gold 已有: {len(skip_already)}\n")
        f.write(f"- 跳过垃圾: {len(skip_garbage)}\n")
        f.write(f"- 跳过类型冲突: {len(skip_conflict)}\n")
        f.write(f"- **补入: {len(will_add)} 条** → 新 gold 共 {len(gold_rows) + len(will_add)} 条\n\n")
        f.write(f"## 补入条目（全部 {len(will_add)} 条）\n\n")
        for h, t in will_add:
            f.write(f"- ({h}, instance_of, {t})\n")
        f.write(f"\n## 跳过的垃圾 ({len(skip_garbage)} 条)\n\n")
        for (h, rel, t), reason in skip_garbage:
            f.write(f"- ({h}, {rel}, {t})  *{reason}*\n")
        if skip_conflict:
            f.write(f"\n## 类型冲突 ({len(skip_conflict)} 条)\n\n")
            for h, t, gold_t, _ in skip_conflict:
                f.write(f"- pred: ({h}, instance_of, {t})  vs  gold: ({h}, instance_of, {gold_t})\n")
    print(f"[augment] 报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
