"""把候选 JSON（默认 data/entities_to_add_ch4.json）合并进 data/entities_by_type.json。

- 自动备份原文件到 data/entities_by_type.backup_<timestamp>.json
- 跳过 _DROPPED_NUMERIC_
- 同实体可能在多个 type 中重复（理论上不会），按"先到优先"处理
- 新词条加到对应 type 列表末尾，避免破坏原有顺序

用法::

    uv run python tools/merge_ner_terms.py                                    # 合并 ch4 候选
    uv run python tools/merge_ner_terms.py --add data/entities_to_add_llm_audited.json
    uv run python tools/merge_ner_terms.py --add path/to/your.json --src data/entities_by_type.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="把候选 JSON 合并进 entities_by_type.json")
    parser.add_argument(
        "--src",
        default=os.path.join(ROOT, "data", "entities_by_type.json"),
        help="目标词典路径（会备份+原地写回）",
    )
    parser.add_argument(
        "--add",
        default=os.path.join(ROOT, "data", "entities_to_add_ch4.json"),
        help="候选 JSON 路径（格式：{type: [{term, ...}, ...]}）",
    )
    args = parser.parse_args()

    src_path = args.src
    add_path = args.add

    if not os.path.exists(src_path):
        print(f"[merge] 找不到 {src_path}")
        return 1
    if not os.path.exists(add_path):
        print(f"[merge] 找不到 {add_path}，请先运行 tools/suggest_ner_terms.py")
        return 1

    with open(src_path, encoding="utf-8-sig") as f:
        existing: dict[str, list[str]] = json.load(f)
    with open(add_path, encoding="utf-8-sig") as f:
        additions: dict[str, list[dict]] = json.load(f)

    # 备份
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(ROOT, "data", f"entities_by_type.backup_{ts}.json")
    shutil.copyfile(src_path, backup)
    print(f"[merge] 已备份: {backup}")

    # 现有词典中已存在的实体集合（用于跨 type 去重）
    existing_terms: set[str] = set()
    for v in existing.values():
        for w in v:
            existing_terms.add((w or "").strip())

    added_total = 0
    per_type_added: dict[str, int] = {}
    for etype, items in additions.items():
        if etype == "_DROPPED_NUMERIC_":
            continue
        if not isinstance(items, list):
            continue
        if etype not in existing:
            existing[etype] = []
        new_terms = []
        for item in items:
            term = item.get("term") if isinstance(item, dict) else None
            if not term:
                continue
            term = term.strip()
            if not term or term in existing_terms:
                continue
            new_terms.append(term)
            existing_terms.add(term)
        if new_terms:
            existing[etype].extend(new_terms)
            per_type_added[etype] = len(new_terms)
            added_total += len(new_terms)

    with open(src_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[merge] 合并完成：新增 {added_total} 个实体")
    for et, n in sorted(per_type_added.items()):
        print(f"  {et:24s} +{n}  (现有 {len(existing[et])})")
    print(f"[merge] 写回: {src_path}")
    print(f"[merge] 备份:   {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
