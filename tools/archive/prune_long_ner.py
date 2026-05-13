"""反向裁剪：从 entities_by_type.json 中删除"长复合实体"以恢复 NER 短词识别。

启发式规则：
    - 仅考虑长度 >= MIN_LEN 的候选（默认 8 字）；
    - 如果实体字符串里**包含**词典里现有的 >=2 字短实体（且该短实体非纯数字/英文），
      则视为"长复合实体"，可被裁剪；
    - 保留规则（不裁剪）：
        * 字母/数字/连字符开头：NACA0008 / Boeing106 / ACADO / NextGen / RQ-4 ...
        * 所属类型在 KEEP_TYPES 内（AIRCRAFT / EQUATION / ORGANIZATION 偏专名）；
        * 显式白名单 KEEP_TERMS（领域核心术语，例如 "大展弦比单翼构型"）。

用法::
    uv run python tools/prune_long_ner.py --dry-run        # 仅列出待删
    uv run python tools/prune_long_ner.py --apply          # 真正写回（自动备份）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_LEN = 8
ALNUM_PREFIX = re.compile(r"^[A-Za-z0-9]")
DIGITS_ONLY = re.compile(r"^[\d.]+$")

# 这些类型默认保留长实体（多为专有名词）。
KEEP_TYPES: set[str] = {
    "AIRCRAFT",      # 型号名（"波音777-200ER" 之类）
    "EQUATION",      # 命名方程
    "ORGANIZATION",  # 公司/机构
    "PERSON",
    "MATERIAL",      # 材料牌号
}

# 即使长且复合，也强制保留的核心术语
KEEP_TERMS: set[str] = {
    "大展弦比单翼构型",  # 论文核心构型，保留
    "大展弦比平直机翼",
    "大展弦比梯形翼",
    "单-双折叠翼变体布局",
}


def load_dict(path: str) -> dict[str, list[str]]:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def build_short_index(d: dict[str, list[str]]) -> list[str]:
    """收集所有 >=2 字、<MIN_LEN 字、非纯数字/纯英文的短实体。"""
    out: set[str] = set()
    for vs in d.values():
        for t in vs:
            t = (t or "").strip()
            if not t:
                continue
            if len(t) < 2 or len(t) >= MIN_LEN:
                continue
            if DIGITS_ONLY.match(t):
                continue
            if ALNUM_PREFIX.match(t):
                # 短英文专名（如 Gap / Stagger）也跳过，避免误吞
                continue
            out.add(t)
    return sorted(out, key=len, reverse=True)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    p = argparse.ArgumentParser(description="反向裁剪长复合 NER 实体")
    p.add_argument("--src", default=os.path.join(ROOT, "data", "entities_by_type.json"))
    p.add_argument("--min-len", type=int, default=MIN_LEN, help=f"长度阈值（默认 {MIN_LEN}）")
    p.add_argument("--dry-run", action="store_true", help="仅打印待删，不写回")
    p.add_argument("--apply", action="store_true", help="确认写回（与 --dry-run 互斥）")
    p.add_argument(
        "--only-from",
        default="",
        help="只在该候选 JSON 列表内裁剪（如 data/entities_to_add_llm_audited.json），"
        "不动 base 词典里原有的长实体",
    )
    args = p.parse_args()

    if not args.dry_run and not args.apply:
        print("[prune] 未指定 --dry-run 或 --apply，默认 --dry-run")
        args.dry_run = True

    src_path = args.src
    d = load_dict(src_path)
    short_index = build_short_index(d)
    print(f"[prune] 加载词典: {sum(len(v) for v in d.values())} 实体")
    print(f"[prune] 用作吞并证据的短实体（<{args.min_len}字）共 {len(short_index)} 个")

    only_from_terms: set[str] = set()
    if args.only_from:
        with open(args.only_from, encoding="utf-8-sig") as f:
            raw = json.load(f)
        for items in raw.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict):
                    term = (it.get("term") or "").strip()
                elif isinstance(it, str):
                    term = it.strip()
                else:
                    term = ""
                if term:
                    only_from_terms.add(term)
        print(f"[prune] --only-from 启用：仅裁剪 {len(only_from_terms)} 个候选中的长实体\n")
    else:
        print("")

    # 找候选
    to_drop: list[tuple[str, str, list[str]]] = []  # (etype, term, hit_short_terms)
    keep_long_no_evidence: list[tuple[str, str]] = []

    for etype, vs in d.items():
        if etype in KEEP_TYPES:
            continue
        for term in list(vs):
            t = (term or "").strip()
            if len(t) < args.min_len:
                continue
            if t in KEEP_TERMS:
                continue
            if ALNUM_PREFIX.match(t):
                continue  # 专名前缀，跳过
            if only_from_terms and t not in only_from_terms:
                continue  # 不在候选清单中，跳过（不动 base 词典）

            hits: list[str] = []
            for s in short_index:
                if len(s) <= 1:
                    continue
                if s == t:
                    continue
                if s in t:
                    hits.append(s)
                    if len(hits) >= 4:
                        break

            if hits:
                to_drop.append((etype, t, hits))
            else:
                keep_long_no_evidence.append((etype, t))

    print(f"[prune] 候选删除（长复合实体）：{len(to_drop)} 条")
    print(f"[prune] 保留（长但无短子串证据）：{len(keep_long_no_evidence)} 条\n")

    print("=== 待删除（按类型排序） ===")
    by_type: dict[str, list[tuple[str, list[str]]]] = {}
    for etype, term, hits in to_drop:
        by_type.setdefault(etype, []).append((term, hits))
    for etype in sorted(by_type):
        print(f"\n[{etype}] {len(by_type[etype])} 条")
        for term, hits in by_type[etype]:
            print(f"  - {term:30s}  ← 含短词: {', '.join(hits[:3])}")

    if args.dry_run:
        print("\n[prune] dry-run 完成。如需写回，使用 --apply。")
        return 0

    # 写回
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(ROOT, "data", f"entities_by_type.backup_{ts}.json")
    shutil.copyfile(src_path, backup)
    print(f"\n[prune] 已备份: {backup}")

    drop_set: set[tuple[str, str]] = {(et, term) for et, term, _ in to_drop}
    new_d: dict[str, list[str]] = {}
    for etype, vs in d.items():
        new_d[etype] = [t for t in vs if (etype, t) not in drop_set]

    with open(src_path, "w", encoding="utf-8") as f:
        json.dump(new_d, f, ensure_ascii=False, indent=2)

    print(f"[prune] 已写回: {src_path}")
    print(f"[prune] 共删除 {len(to_drop)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
