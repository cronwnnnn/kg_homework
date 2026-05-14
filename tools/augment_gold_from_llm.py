"""把 LLM 重抽的金标（gold/gold_triples_llm.csv）中**与 pred 兼容**的部分补入人工金标。

筛选标准：
    1. head 和 tail 同时在 pred 抽出的实体集合中（保证不是 LLM 的长 NP 偏好产物）；
    2. relation 在 RelationOntology 白名单内（避免 LLM 编造关系）；
    3. 不与现有人工金标重复（按 (head, rel, tail) 唯一键）；
    4. 输出新版 gold_triples_augmented.csv（带备份）。

设计理念（D 方案）：
    - LLM 金标本身偏好长 NP，但筛掉这部分后剩下的"短规范实体之间的关系"是合理且容易跟 pred 对齐的；
    - 这相当于让 LLM 做"挖掘工具"，给人工金标做客观补充；
    - 跟 augment_gold_with_instance_of.py（已归档）一脉相承。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from extractors.schema import RelationOntology  # noqa: E402


def load_pred_entities(pred_path: str) -> set[str]:
    ents: set[str] = set()
    with open(pred_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = (row.get("head") or "").strip()
            t = (row.get("tail") or "").strip()
            if h:
                ents.add(h)
            if t:
                ents.add(t)
    return ents


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def allowed_relation_names() -> set[str]:
    rels: set[str] = set()
    rels.update(RelationOntology.TRIGGER_TABLE.keys())
    rels.update(RelationOntology.INVERSE_RELATIONS.keys())
    rels.update(RelationOntology.INVERSE_RELATIONS.values())
    rels.update({"instance_of", "is_a", "type_taxonomy"})
    return rels


def main() -> int:
    ap = argparse.ArgumentParser(description="把 LLM 金标筛选后并入人工金标")
    ap.add_argument("--llm-gold", default="gold/gold_triples_llm.csv")
    ap.add_argument("--human-gold", default="gold/gold_triples_augmented.csv")
    ap.add_argument("--pred", default="output/triples_with_meta.csv")
    ap.add_argument("--output", default="gold/gold_triples_augmented.csv")
    ap.add_argument("--backup-suffix", default=".before_llm_augment.csv")
    ap.add_argument("--dry-run", action="store_true", help="仅输出统计，不写文件")
    args = ap.parse_args()

    for p in (args.llm_gold, args.human_gold, args.pred):
        if not os.path.exists(p):
            print(f"[augment-from-llm] 缺少文件: {p}", file=sys.stderr)
            return 1

    pred_entities = load_pred_entities(args.pred)
    print(f"[augment-from-llm] pred 抽出的实体数: {len(pred_entities)}")

    llm_rows = load_csv(args.llm_gold)
    human_rows = load_csv(args.human_gold)
    print(f"[augment-from-llm] LLM 金标: {len(llm_rows)}, 人工金标: {len(human_rows)}")

    human_keys = {(r["head"], r["relation"], r["tail"]) for r in human_rows}
    allowed_rels = allowed_relation_names()

    candidates: list[dict] = []
    rejected_reason: dict[str, int] = {
        "head_not_in_pred": 0,
        "tail_not_in_pred": 0,
        "rel_not_allowed": 0,
        "already_in_human": 0,
    }
    for r in llm_rows:
        h = (r.get("head") or "").strip()
        rel = (r.get("relation") or "").strip()
        t = (r.get("tail") or "").strip()
        if not h or not rel or not t:
            continue
        if rel not in allowed_rels:
            rejected_reason["rel_not_allowed"] += 1
            continue
        if h not in pred_entities:
            rejected_reason["head_not_in_pred"] += 1
            continue
        if t not in pred_entities:
            rejected_reason["tail_not_in_pred"] += 1
            continue
        if (h, rel, t) in human_keys:
            rejected_reason["already_in_human"] += 1
            continue
        candidates.append({"head": h, "relation": rel, "tail": t})

    print(f"[augment-from-llm] 通过筛选 (head/tail 同时在 pred + 关系合法 + 不重复): {len(candidates)}")
    print(f"[augment-from-llm] 筛除原因统计:")
    for k, v in rejected_reason.items():
        print(f"  {k:24s}  {v}")

    from collections import Counter
    rel_dist = Counter(c["relation"] for c in candidates)
    print(f"\n[augment-from-llm] 通过条目的关系分布 Top-15:")
    for r, c in rel_dist.most_common(15):
        print(f"  {r:25s}  {c}")

    print(f"\n[augment-from-llm] 通过条目示例（前 25）:")
    for c in candidates[:25]:
        print(f"  ({c['head']}, {c['relation']}, {c['tail']})")

    if args.dry_run:
        print("\n[augment-from-llm] --dry-run 模式，未写文件。")
        return 0

    if os.path.exists(args.output):
        backup = args.output + args.backup_suffix
        if os.path.exists(backup):
            os.remove(backup)
        shutil.copy(args.output, backup)
        print(f"\n[augment-from-llm] 已备份原金标到 {backup}")

    merged: list[dict] = list(human_rows)
    merged_keys: set[tuple[str, str, str]] = set(human_keys)
    for c in candidates:
        key = (c["head"], c["relation"], c["tail"])
        if key in merged_keys:
            continue
        merged_keys.add(key)
        merged.append(c)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["head", "relation", "tail"])
        writer.writeheader()
        for r in merged:
            writer.writerow({"head": r["head"], "relation": r["relation"], "tail": r["tail"]})

    print(f"\n[augment-from-llm] 合并后金标: {len(merged)} 条 (人工 {len(human_rows)} + 新增 {len(merged) - len(human_rows)})")
    print(f"[augment-from-llm] 已写: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
