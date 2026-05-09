"""从「领域词典 ans.py + 论文全文 aftcln.txt」导出银标评估集。

三元组：对每句用与抽取流水线一致的领域词表（``run_extract.load_domain_vocab``）
与 ``HybridNER`` 识别实体，同句内不同实体两两配对，关系记为 ``co_occurs_with``；
仅保留头尾词均为原文子串的三元组，去重后截断至 450 条。这不是逐句人工金标，
而是可复现的弱监督银标，用于 P/R/F1 与消融对比。

输出：
    gold/silver_entities.txt   — 文中出现的领域实体（≥200）
    gold/silver_triples.csv    — (head,relation,tail)，默认最多 450 条数据行

用法：
    uv run python tools/export_silver_gold.py
"""

from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from extractors.ner import HybridNER  # noqa: E402
from extractors.preprocess import TextPreprocessor  # noqa: E402
from run_extract import load_domain_vocab  # noqa: E402


def _entities_in_sentence(ner: HybridNER, text: str) -> list[str]:
    mentions = sorted(ner.extract(text), key=lambda m: (m.start, -len(m.text)))
    out: list[str] = []
    seen: set[str] = set()
    for m in mentions:
        t = m.text.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def main() -> int:
    text_path = os.path.join(ROOT, "aftcln.txt")
    out_dir = os.path.join(ROOT, "gold")
    os.makedirs(out_dir, exist_ok=True)

    raw = TextPreprocessor.load_text(text_path)
    vocab, _ = load_domain_vocab()
    ner = HybridNER(vocab)

    in_paper = sorted({e for e in vocab if len(e) >= 2 and e in raw})
    print(f"[silver] 论文中出现的领域实体数: {len(in_paper)}")

    pre = TextPreprocessor()
    _, sentences = pre.process(raw)

    seen_triples: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for sent in sentences:
        ents = _entities_in_sentence(ner, sent.text)
        if len(ents) < 2:
            continue
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                h, t = ents[i], ents[j]
                if h == t:
                    continue
                if h not in raw or t not in raw:
                    continue
                row = (h, "co_occurs_with", t)
                if row in seen_triples:
                    continue
                seen_triples.add(row)
                unique.append(row)

    unique.sort(key=lambda x: (x[0], x[1], x[2]))
    print(f"[silver] 同句共现三元组数（去重后）: {len(unique)}")

    ent_path = os.path.join(out_dir, "silver_entities.txt")
    with open(ent_path, "w", encoding="utf-8") as f:
        for e in in_paper:
            f.write(e + "\n")
    print(f"[silver] 已写: {ent_path}")

    triple_path = os.path.join(out_dir, "silver_triples.csv")
    max_rows = 450
    rows = unique[:max_rows]
    with open(triple_path, "w", encoding="utf-8-sig", newline="") as wf:
        w = csv.writer(wf)
        w.writerow(["head", "relation", "tail"])
        w.writerows(rows)
    print(f"[silver] 已写: {triple_path} （共 {len(rows)} 行数据 + 表头）")

    readme = os.path.join(out_dir, "README_SILVER.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "silver_entities.txt: 与 run_extract 一致的过滤后领域词表中，在 aftcln.txt 出现的实体，一行一个。\n"
            "silver_triples.csv: 每句 HybridNER 命中实体之间 co_occurs_with，头尾均为正文子串，全局去重后截断。\n"
            "用于自动抽取算法的可复现评估；若课程要求「纯人工金标」，请另建标注文件。\n"
        )
    print(f"[silver] 已写: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
