"""将自动抽取结果与金标/银标 CSV 对比，计算实体与三元组层面的 P/R/F1。

默认预测：output/triples_with_meta.csv
默认参考：gold/silver_triples.csv（由 tools/export_silver_gold.py 生成）

用法：
    uv run python tools/export_silver_gold.py   # 先生成银标
    uv run python evaluate_kg.py
    uv run python evaluate_kg.py --pred output/triples_with_meta.csv --gold my_gold.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable


def _read_triples_csv(path: str) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # 兼容中英表头
        def pick(row: dict[str, str], *keys: str) -> str:
            for k in keys:
                if k in row and row[k]:
                    return (row[k] or "").strip()
            return ""

        for row in reader:
            h = pick(row, "head", "头实体", "source", "h")
            r = pick(row, "relation", "关系", "type", "r")
            t = pick(row, "tail", "尾实体", "target", "t")
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


def prf1(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="知识图谱自动抽取 vs 银标/金标 评估")
    p.add_argument("--pred", default="output/triples_with_meta.csv", help="预测三元组 CSV")
    p.add_argument("--gold", default="gold/silver_triples.csv", help="金标/银标三元组 CSV")
    p.add_argument(
        "--gold-entities",
        default="gold/silver_entities.txt",
        help="可选：实体银标列表（一行一个）；缺省则用金标三元组端点并集",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.pred):
        print(f"[eval] 找不到预测文件: {args.pred}", file=sys.stderr)
        return 1
    if not os.path.exists(args.gold):
        print(f"[eval] 找不到金标文件: {args.gold}", file=sys.stderr)
        print("[eval] 请先运行: uv run python tools/export_silver_gold.py", file=sys.stderr)
        return 1

    pred_triples = _read_triples_csv(args.pred)
    gold_triples = _read_triples_csv(args.gold)

    if os.path.isfile(args.gold_entities):
        gold_entities = _read_entities_txt(args.gold_entities)
    else:
        gold_entities = _entities_from_triples(gold_triples)

    pred_entities = _entities_from_triples(pred_triples)

    tp_t = pred_triples & gold_triples
    p_t, r_t, f1_t = prf1(pred_triples, gold_triples)
    p_e, r_e, f1_e = prf1(pred_entities, gold_entities)

    report_path = os.path.join("output", "eval_report.txt")
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    lines = [
        "==== 知识图谱抽取评估 ====",
        f"预测文件: {args.pred}",
        f"金标文件: {args.gold}",
        "",
        f"预测三元组数: {len(pred_triples)}",
        f"金标三元组数: {len(gold_triples)}",
        f"三元组 TP: {len(tp_t)}",
        f"三元组 Precision: {p_t:.4f}",
        f"三元组 Recall:    {r_t:.4f}",
        f"三元组 F1:        {f1_t:.4f}",
        "",
        f"预测实体数(端点): {len(pred_entities)}",
        f"金标实体数:       {len(gold_entities)}",
        f"实体 Precision: {p_e:.4f}",
        f"实体 Recall:    {r_e:.4f}",
        f"实体 F1:        {f1_e:.4f}",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[eval] 已写: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
