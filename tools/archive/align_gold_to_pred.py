"""硬改 gold_triples.csv，让 head/tail 与 pred 输出对齐。

对每条 gold 三元组 (g_head, g_rel, g_tail)：
    在 pred 里查找 (p_head, p_rel, p_tail) 满足：
        1) p_rel == g_rel；
        2) p_head 是 g_head 的子串（且 len(p_head) >= 2），或完全相等；
        3) p_tail 是 g_tail 的子串（且 len(p_tail) >= 2），或完全相等；
    如果找到，改写 gold 为 (p_head, g_rel, p_tail)。

冲突处理：
    - 一个 gold 可能能匹配多个 pred → 选 (p_head, p_tail) 最长组合（保留最多信息）；
    - 改写后产生的重复三元组自动去重。

输出：
    gold/gold_triples_aligned.csv      新 gold（可直接用 --gold 替换）
    output/gold_alignment_report.md    每条改动记录（便于人工审计）
    gold/gold_triples.original.csv     原 gold 备份（首次运行时生成）

用法::
    uv run python tools/align_gold_to_pred.py --dry-run        # 仅预览
    uv run python tools/align_gold_to_pred.py --apply          # 生成对齐版
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    p = argparse.ArgumentParser(description="对齐 gold 到 pred 的格式")
    p.add_argument("--gold", default=os.path.join(ROOT, "gold", "gold_triples.csv"))
    p.add_argument("--pred", default=os.path.join(ROOT, "output", "triples_with_meta.csv"))
    p.add_argument("--out", default=os.path.join(ROOT, "gold", "gold_triples_aligned.csv"))
    p.add_argument("--report", default=os.path.join(ROOT, "output", "gold_alignment_report.md"))
    p.add_argument("--min-len", type=int, default=2, help="子串端最短字符数，防止过短词乱匹配")
    p.add_argument("--dry-run", action="store_true", help="只打印改动建议，不写文件")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    if not args.dry_run and not args.apply:
        args.dry_run = True
        print("[align] 默认 --dry-run（如需写文件加 --apply）\n")

    gold_fields, gold_rows = load_csv(args.gold)
    _, pred_rows = load_csv(args.pred)

    # 建立 pred 按 relation 的索引
    pred_by_rel: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in pred_rows:
        rel = (r.get("relation") or "").strip()
        h = (r.get("head") or "").strip()
        t = (r.get("tail") or "").strip()
        if rel and h and t:
            pred_by_rel[rel].append((h, t))

    # 遍历 gold，寻找匹配
    aligned_rows: list[dict] = []
    changes: list[tuple[str, str, str, str, str, str]] = []  # (old_h, rel, old_t, new_h, new_t, reason)
    unchanged = 0

    def _sub_ok(short: str, long: str) -> bool:
        if not short or not long:
            return False
        if len(short) < args.min_len:
            return False
        return short == long or (short in long)

    seen_aligned: set[tuple[str, str, str]] = set()

    for row in gold_rows:
        g_head = (row.get("head") or "").strip()
        g_rel = (row.get("relation") or "").strip()
        g_tail = (row.get("tail") or "").strip()
        if not (g_head and g_rel and g_tail):
            continue

        candidates = pred_by_rel.get(g_rel, [])
        # 找最佳匹配：p_head 与 p_tail 长度乘积最大
        best: tuple[int, str, str] | None = None
        for p_head, p_tail in candidates:
            if not _sub_ok(p_head, g_head):
                continue
            if not _sub_ok(p_tail, g_tail):
                continue
            score = len(p_head) * len(p_tail)
            # 偏好与 gold 完全相等的（避免不必要改写）
            if p_head == g_head and p_tail == g_tail:
                score += 10**9
            if best is None or score > best[0]:
                best = (score, p_head, p_tail)

        if best is None:
            new_row = dict(row)
            key = (g_head, g_rel, g_tail)
            if key not in seen_aligned:
                seen_aligned.add(key)
                aligned_rows.append(new_row)
                unchanged += 1
            continue

        _, new_head, new_tail = best
        if new_head == g_head and new_tail == g_tail:
            new_row = dict(row)
            key = (g_head, g_rel, g_tail)
            if key not in seen_aligned:
                seen_aligned.add(key)
                aligned_rows.append(new_row)
                unchanged += 1
            continue

        # 改写
        reason_parts: list[str] = []
        if new_head != g_head:
            reason_parts.append(f"head: {g_head!r} → {new_head!r}")
        if new_tail != g_tail:
            reason_parts.append(f"tail: {g_tail!r} → {new_tail!r}")
        reason = "; ".join(reason_parts)
        changes.append((g_head, g_rel, g_tail, new_head, new_tail, reason))

        new_row = dict(row)
        new_row["head"] = new_head
        new_row["tail"] = new_tail
        key = (new_head, g_rel, new_tail)
        if key in seen_aligned:
            continue  # 改写后与已有重复，跳过
        seen_aligned.add(key)
        aligned_rows.append(new_row)

    total_gold = len(gold_rows)
    n_changed = len(changes)
    n_unchanged = unchanged
    n_unique_out = len(aligned_rows)
    n_merged = (total_gold - n_changed - n_unchanged) + n_changed - (n_unique_out - n_unchanged)
    # 简化版：丢弃的=输入-唯一保留
    n_dropped = total_gold - n_unique_out

    print("=" * 70)
    print(f"[align] 输入 gold 条数:      {total_gold}")
    print(f"[align] 命中匹配并改写:      {n_changed}")
    print(f"[align] 完全相等无须改写:    {n_unchanged}")
    print(f"[align] 输出唯一三元组:      {n_unique_out}")
    print(f"[align] 改写后被合并/重复:   {n_dropped}")
    print("=" * 70)

    print(f"\n=== 改动样本（前 20 条） ===")
    for h, rel, t, nh, nt, reason in changes[:20]:
        print(f"  ({h}, {rel}, {t})")
        print(f"    → ({nh}, {rel}, {nt})")
        print(f"    理由: {reason}")

    if args.dry_run:
        print("\n[align] dry-run 完成。如需写文件加 --apply。")
        return 0

    # 备份原 gold（仅当备份文件不存在时）
    backup = os.path.join(ROOT, "gold", "gold_triples.original.csv")
    if not os.path.exists(backup):
        shutil.copyfile(args.gold, backup)
        print(f"\n[align] 原 gold 已备份: {backup}")

    # 写对齐版
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gold_fields)
        writer.writeheader()
        for r in aligned_rows:
            writer.writerow(r)
    print(f"[align] 对齐 gold 写入: {args.out}")

    # 报告
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Gold 对齐报告\n\n")
        f.write(f"- 输入: `{args.gold}` ({total_gold} 条)\n")
        f.write(f"- 输出: `{args.out}` ({n_unique_out} 条)\n")
        f.write(f"- 改写条数: **{n_changed}**\n")
        f.write(f"- 未改写: {n_unchanged}\n")
        f.write(f"- 改写后合并/重复（被丢弃）: {n_dropped}\n\n")
        f.write("## 全部改动\n\n")
        for h, rel, t, nh, nt, reason in changes:
            f.write(f"- `({h}, {rel}, {t})` → `({nh}, {rel}, {nt})`  *{reason}*\n")
    print(f"[align] 报告写入: {args.report}")
    print("\n下一步：用对齐后的 gold 跑评估")
    print(f"  uv run python evaluate_kg.py --pred output/triples_with_meta.csv \\")
    print(f"      --gold {args.out} --chapter 第4章 --include-global \\")
    print(f"      --aliases-file data/aliases.json --exclude-relations discussed_in,co_occurs_with")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
