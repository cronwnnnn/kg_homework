"""智能过滤 pred CSV 中的低价值 instance_of 三元组。

删除规则（仅作用于 relation == 'instance_of' 的行）：
    R1: head 纯数字 / 含 % / 含"倍" / 数字开头
    R2: head 长度 < 2 或 >= 8 字（过短或过长复合）
    R3: head 含连字符且非字母开头（如 "10-15%"）
    R4: head 含百分号、单位 m/km/kg/N·m
    R5: head 与 tail 中文同义（如 ("飞行器", "飞行器")）

用法::
    uv run python tools/filter_instance_of.py
    uv run python tools/filter_instance_of.py --dry-run     # 仅打印将删的
    uv run python tools/filter_instance_of.py --max-head-len 10  # 调整长度上限
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RE_VALUE_LIKE = re.compile(
    r"(^[0-9.]+$)|(%)|(倍)|(^[\dA-Za-z][-0-9.]*$)|(m/s)|(N·m)|(°)|(_)|(\.\.)"
)


def is_low_value_head(head: str, max_len: int = 8) -> tuple[bool, str]:
    h = (head or "").strip()
    if not h:
        return True, "空字符串"
    if len(h) < 2:
        return True, "<2字"
    if len(h) >= max_len:
        return True, f">={max_len}字（过长复合）"
    if h.isdigit():
        return True, "纯数字"
    if h[0].isdigit():
        return True, "数字开头"
    if _RE_VALUE_LIKE.search(h):
        return True, "含 %/倍/单位/下划线 等数值标记"
    return False, ""


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="过滤 pred 中的低价值 instance_of")
    parser.add_argument("--src", default=os.path.join(ROOT, "output", "triples_with_meta.csv"))
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "output", "triples_with_meta.filtered.csv"),
        help="输出过滤后的 CSV（不覆盖原文件）",
    )
    parser.add_argument("--max-head-len", type=int, default=8, help="head 长度上限")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        all_rows = list(reader)

    total = len(all_rows)
    kept: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    for r in all_rows:
        if r.get("relation") != "instance_of":
            kept.append(r)
            continue
        head = r.get("head", "")
        bad, reason = is_low_value_head(head, args.max_head_len)
        # 额外规则：head == tail
        if not bad and head == r.get("tail", ""):
            bad, reason = True, "head==tail"
        if bad:
            dropped.append((r, reason))
        else:
            kept.append(r)

    reason_counter: dict[str, int] = {}
    for _, reason in dropped:
        reason_counter[reason] = reason_counter.get(reason, 0) + 1

    print(f"[filter] 总三元组: {total}")
    print(f"[filter] 保留:    {len(kept)}")
    print(f"[filter] 删除 instance_of: {len(dropped)}")
    print(f"\n[filter] 删除原因分布:")
    for reason, n in sorted(reason_counter.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}  {n}")

    print(f"\n[filter] 删除样本（前 20）:")
    for r, reason in dropped[:20]:
        print(f"  [{reason}]  ({r.get('head','')}, instance_of, {r.get('tail','')})")

    if args.dry_run:
        print("\n[filter] dry-run 完成。如需写文件去掉 --dry-run。")
        return 0

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"\n[filter] 写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
