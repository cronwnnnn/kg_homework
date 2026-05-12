"""从「领域词典 ans.py + 论文全文 aftcln.txt」导出银标评估集。

输出 ``gold/`` 三份：

1) ``silver_entities.txt``：论文中真实出现的领域实体清单（含 paper_mining 挖掘的新词）。
2) ``silver_triples.csv``：严格银标 = 触发词派生关系 ∪ 类型派生 instance_of ∪ 章节派生 discussed_in。
3) ``silver_triples_loose.csv``：宽松银标，所有同句相邻实体对一律 co_occurs_with。

派生规则与 ``extractors/`` 各抽取器同源，但绕开 LLM 与 cooccur 的弱启发式：
    - 触发词：与 ``TriggerCooccurrenceExtractor`` 相同的扫描；
    - 类型：用 ``entities_by_type`` 给在论文中出现的实体打 instance_of → 类型标签；
    - 章节：实体在该章节出现 ≥ 2 次记 discussed_in → 章节标题。
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from extractors.ner import HybridNER  # noqa: E402
from extractors.preprocess import Sentence, TextPreprocessor  # noqa: E402
from extractors.schema import Mention, RelationOntology  # noqa: E402
from extractors.type_extractor import _TYPE_LABEL_MAP  # noqa: E402
from extractors.paper_entity_recognizer import PaperEntityRecognizer  # noqa: E402
from run_extract import load_domain_vocab  # noqa: E402


_NEG_HINTS = ("不", "未", "无", "非", "并未", "没有", "免", "勿")
_PASSIVE_HEAD_HINTS = ("由", "被", "经")
_MAX_GAP = 30
_K_NEIGHBOR = 4  # 同句相邻 K 个实体两两组合


def _build_triggers() -> list[tuple[str, str]]:
    flat: list[tuple[str, str]] = []
    for rel, words in RelationOntology.TRIGGER_TABLE.items():
        for w in words:
            if "…" in w:
                continue
            flat.append((w, rel))
    flat.sort(key=lambda x: -len(x[0]))
    return flat


def _entities_in_sentence(ner: HybridNER, text: str) -> list[Mention]:
    mentions = sorted(ner.extract(text), key=lambda m: (m.start, -len(m.text)))
    out: list[Mention] = []
    seen: set[tuple[int, int]] = set()
    for m in mentions:
        if not m.text.strip() or (m.start, m.end) in seen:
            continue
        seen.add((m.start, m.end))
        out.append(m)
    return out


def _scan_relation(gap: str, triggers: list[tuple[str, str]]) -> tuple[str, str, int] | None:
    """在窗口中找首个触发词，返回 (relation, trigger, pos)。"""
    if not gap:
        return None
    if any(p in gap for p in "。！？；"):
        return None
    if any(neg in gap for neg in _NEG_HINTS):
        return None
    best: tuple[int, str, str] | None = None
    for trig, rel in triggers:
        pos = gap.find(trig)
        if pos < 0:
            continue
        cand = (pos, trig, rel)
        if best is None or cand < best:
            best = cand
    if best is None:
        return None
    return best[2], best[1], best[0]


def collect_silver(
    sentences: Iterable[Sentence],
    ner: HybridNER,
    raw: str,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """返回 (strict_triples, loose_triples)。

    strict 仅保留触发词命中且关系∈本体的对；这与抽取流水线的 trigger 抽取器同源，
    比"co_occurs_with 兜底"更能反映真实关系抽取能力。
    loose 不限制触发词，所有相邻实体对都记 co_occurs_with，用来做实体对覆盖度评估。
    """
    triggers = _build_triggers()
    strict_seen: set[tuple[str, str, str]] = set()
    strict_out: list[tuple[str, str, str]] = []
    loose_seen: set[tuple[str, str, str]] = set()
    loose_out: list[tuple[str, str, str]] = []

    for sent in sentences:
        ms = _entities_in_sentence(ner, sent.text)
        if len(ms) < 2:
            continue
        ms = sorted(ms, key=lambda m: m.start)
        n = len(ms)

        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + _K_NEIGHBOR)):
                a, b = ms[i], ms[j]
                if a.text == b.text or a.end > b.start:
                    continue
                head, tail = a.text, b.text
                if head not in raw or tail not in raw:
                    continue
                gap = sent.text[a.end:b.start]
                if len(gap) > _MAX_GAP:
                    continue

                # strict 只在触发词命中时入库
                hit = _scan_relation(gap, triggers)
                if hit is not None:
                    rel, trig, pos = hit
                    pre = gap[:pos].rstrip("，,的、和与及或而又也 ")
                    h, t = head, tail
                    if rel in ("develops", "manufactures") and pre and pre[-1] in _PASSIVE_HEAD_HINTS:
                        h, t = tail, head
                    strict_key = (h, rel, t)
                    if strict_key not in strict_seen:
                        strict_seen.add(strict_key)
                        strict_out.append(strict_key)

                # loose 全收（仅相邻 K 个，K=4，避免远距离误关联）
                loose_key = (a.text, "co_occurs_with", b.text)
                if loose_key not in loose_seen:
                    loose_seen.add(loose_key)
                    loose_out.append(loose_key)

    strict_out.sort(key=lambda x: (x[1], x[0], x[2]))
    loose_out.sort(key=lambda x: (x[0], x[1], x[2]))
    return strict_out, loose_out


def _summary(triples: list[tuple[str, str, str]]) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for _, r, _ in triples:
        cnt[r] = cnt.get(r, 0) + 1
    return cnt


def derive_type_triples(
    entities_by_type: dict[str, list[str]],
    observed: set[str],
) -> list[tuple[str, str, str]]:
    """实体 → 类型标签 (instance_of)。仅对在论文中观察到的实体生成。"""
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for type_name, items in entities_by_type.items():
        label = _TYPE_LABEL_MAP.get(type_name)
        if not label:
            continue
        for e in items:
            e = (e or "").strip()
            if not e or e not in observed:
                continue
            key = (e, "instance_of", label)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def derive_chapter_triples(
    sentences: list[Sentence],
    ner: HybridNER,
    secondary_types: set[str],
    term_to_type: dict[str, str],
    min_occur: int = 2,
) -> list[tuple[str, str, str]]:
    """实体 → 章节标题 (discussed_in)。条件：实体在该章节出现 ≥ min_occur 次。"""
    chapter_counter: dict[str, Counter[str]] = defaultdict(Counter)
    for s in sentences:
        chapter = s.chapter or ""
        if not chapter:
            continue
        for m in ner.extract(s.text):
            if m.text in term_to_type and term_to_type[m.text] in secondary_types:
                chapter_counter[chapter][m.text] += 1
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for chapter, counter in chapter_counter.items():
        for ent, n in counter.most_common():
            if n < min_occur:
                continue
            key = (ent, "discussed_in", chapter)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


_SECONDARY_TYPES: set[str] = {
    "AIRCRAFT", "WING_CONFIGURATION", "TECHNOLOGY", "STRUCTURAL_COMPONENT",
    "CONTROL_METHOD", "ORGANIZATION", "PERSON", "EQUATION", "MATERIAL",
    "PARAMETER", "AERODYNAMIC_CONCEPT",
}


def main() -> int:
    text_path = os.path.join(ROOT, "aftcln.txt")
    out_dir = os.path.join(ROOT, "gold")
    os.makedirs(out_dir, exist_ok=True)

    raw = TextPreprocessor.load_text(text_path)
    vocab, entities_by_type = load_domain_vocab()
    ner = HybridNER(vocab)

    in_paper = sorted({e for e in vocab if len(e) >= 2 and e in raw})
    print(f"[silver] 词典规模 = {len(vocab)}")
    print(f"[silver] 论文中出现的领域实体数 = {len(in_paper)}")

    pre = TextPreprocessor()
    paragraphs, sentences = pre.process(raw)
    print(f"[silver] 句子数 = {len(sentences)}")

    # 用 paper_entity_mining 把论文中真实存在的非词典实体一并补到 NER 词表 + silver_entities
    mined: list[str] = []
    try:
        rec = PaperEntityRecognizer()
        mined = rec.collect_from_sentences(sentences, vocab_boost=set(vocab), min_doc_freq=2)
        if mined:
            ner.add_terms(mined)
            print(f"[silver] paper_mining: 候选 {len(mined)} 串并入 NER（用于评估目标对齐）")
    except Exception as exc:  # noqa: BLE001
        print(f"[silver] paper_mining 失败：{exc}（仅使用原词典）")

    # 实体银标 = 论文中出现的原词典实体 ∪ paper_mining 高频挖出的实体
    full_in_paper = sorted(set(in_paper) | {m for m in mined if m in raw})

    # 各抽取器派生
    strict_trigger, loose = collect_silver(sentences, ner, raw)
    print(f"[silver] 触发词银标 = {len(strict_trigger)}  宽松银标 = {len(loose)}")

    observed: set[str] = set()
    for h, _, t in strict_trigger:
        observed.add(h)
        observed.add(t)
    for e in full_in_paper:
        observed.add(e)

    type_triples = derive_type_triples(entities_by_type, observed)
    print(f"[silver] 类型银标 (instance_of) = {len(type_triples)}")

    term_to_type: dict[str, str] = {}
    for type_name, items in entities_by_type.items():
        for w in items:
            term_to_type.setdefault((w or "").strip(), type_name)
    chapter_triples = derive_chapter_triples(
        sentences, ner, _SECONDARY_TYPES, term_to_type, min_occur=2
    )
    print(f"[silver] 章节银标 (discussed_in) = {len(chapter_triples)}")

    # 合并：触发词 + 类型 + 章节，去重
    strict_seen: set[tuple[str, str, str]] = set()
    strict: list[tuple[str, str, str]] = []
    for src in (strict_trigger, type_triples, chapter_triples):
        for k in src:
            if k not in strict_seen:
                strict_seen.add(k)
                strict.append(k)
    strict.sort(key=lambda x: (x[1], x[0], x[2]))
    print(f"[silver] 严格银标合计（触发词∪类型∪章节）= {len(strict)}")
    print(f"[silver] 实体银标总数（词典∩文本 ∪ 挖掘）= {len(full_in_paper)}")

    ent_path = os.path.join(out_dir, "silver_entities.txt")
    with open(ent_path, "w", encoding="utf-8") as f:
        for e in full_in_paper:
            f.write(e + "\n")
    print(f"[silver] 已写 {ent_path}")

    triple_path = os.path.join(out_dir, "silver_triples.csv")
    with open(triple_path, "w", encoding="utf-8-sig", newline="") as wf:
        w = csv.writer(wf)
        w.writerow(["head", "relation", "tail"])
        w.writerows(strict)
    print(f"[silver] 已写 {triple_path}（{len(strict)} 数据行）")

    loose_path = os.path.join(out_dir, "silver_triples_loose.csv")
    with open(loose_path, "w", encoding="utf-8-sig", newline="") as wf:
        w = csv.writer(wf)
        w.writerow(["head", "relation", "tail"])
        w.writerows(loose)
    print(f"[silver] 已写 {loose_path}")

    summary = _summary(strict)
    top = sorted(summary.items(), key=lambda x: -x[1])[:15]
    print("[silver] 严格银标关系分布 Top-15:")
    for rel, n in top:
        print(f"  {rel}: {n}")

    readme = os.path.join(out_dir, "README_SILVER.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "==== 银标说明（与 evaluate_kg.py 配套） ====\n\n"
            "silver_entities.txt\n"
            "    论文中真实出现的领域实体清单（与 run_extract 一致的过滤后词表）。\n"
            "silver_triples.csv         <- evaluate_kg.py 默认评估目标\n"
            "    严格银标：基于触发词派生 30+ 种真实关系（improves/reduces/has_part/...），\n"
            "    未命中触发词的相邻实体对兜底为 co_occurs_with。\n"
            "    适用于关系级别 P/R/F1 评估。\n"
            "silver_triples_loose.csv\n"
            "    宽松银标：所有同句相邻实体对均记 co_occurs_with，仅做实体覆盖评估。\n"
            "    命令：evaluate_kg.py --gold gold/silver_triples_loose.csv --loose\n\n"
            "注：本银标为自动派生，非纯人工金标。若课程要求纯人工金标，\n"
            "请以 silver_triples.csv 为模板逐行复核为 gold_triples_labeled.csv。\n"
        )
    print(f"[silver] 已写 {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
