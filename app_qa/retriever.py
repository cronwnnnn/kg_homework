"""检索器：根据 ParsedQuery 在 KGStore + Corpus 中检索三元组与原文证据。

输出 RetrievalResult：
    triples         : 主答案候选三元组（按 score 排序）
    evidence_rows   : 原文证据句子（含 chapter / section）
    subgraph_seed   : 用于绘制子图的中心实体
    related_triples : 辅助显示的同实体其他三元组
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus_index import Corpus, SentenceRecord
from .kg_store import KGStore, TripleRow
from .query_parser import ParsedQuery, QueryIntent


@dataclass
class RetrievalResult:
    intent: QueryIntent
    triples: list[TripleRow] = field(default_factory=list)
    evidence_rows: list[SentenceRecord] = field(default_factory=list)
    subgraph_seed: str = ""
    related_triples: list[TripleRow] = field(default_factory=list)
    note: str = ""


_PROPERTY_RELATIONS = (
    "has_part", "has_parameter", "has_performance", "has_configuration",
    "has_value", "has_property", "uses_material", "made_of",
)
_CAUSE_RELATIONS = ("causes", "reduces", "improves", "affects", "leads_to", "generates")
_EFFECT_RELATIONS = _CAUSE_RELATIONS


def _dedup(rows: list[TripleRow]) -> list[TripleRow]:
    seen: set[tuple[str, str, str]] = set()
    out: list[TripleRow] = []
    for r in rows:
        if r.key in seen:
            continue
        seen.add(r.key)
        out.append(r)
    return out


def retrieve(parsed: ParsedQuery, kg: KGStore, corpus: Corpus) -> RetrievalResult:
    intent = parsed.intent
    ents = parsed.entities
    primary = parsed.primary
    secondary = parsed.secondary

    res = RetrievalResult(intent=intent, subgraph_seed=primary)

    if not primary:
        res.note = "没有从问题中识别到任何领域实体。建议把要查的实体写明确，例如：「全球鹰 是多少」"
        return res

    rows: list[TripleRow] = []
    rel_filter: tuple[str, ...] = ()

    if intent == QueryIntent.DEFINITION:
        rows = kg.out_edges(primary, relation="instance_of") + kg.out_edges(primary, relation="is_a")
        rows = _dedup(rows)
        if not rows:
            for rel in ("has_part", "used_for", "develops", "applied_to"):
                rows.extend(kg.out_edges(primary, relation=rel))
                if rows:
                    break
        res.triples = sorted(rows, key=lambda t: -t.score)[:8]

    elif intent == QueryIntent.LIST_INSTANCE:
        rows = [t for t in kg.triples_with_relation("instance_of") if t.tail == primary]
        if not rows:
            rows = [t for t in kg.triples_with_relation("instance_of") if primary in t.tail]
        res.triples = sorted(rows, key=lambda t: -t.score)[:15]

    elif intent == QueryIntent.VALUE:
        for rel in ("has_value", "greater_than_value", "less_than_value", "equals_to", "approximately"):
            rows.extend(kg.out_edges(primary, relation=rel))
        res.triples = sorted(_dedup(rows), key=lambda t: -t.score)[:8]

    elif intent == QueryIntent.CAUSE:
        rel_filter = (parsed.target_relation,) if parsed.target_relation else _CAUSE_RELATIONS
        for rel in rel_filter:
            rows.extend(kg.in_edges(primary, relation=rel))
        res.triples = sorted(_dedup(rows), key=lambda t: -t.score)[:10]

    elif intent == QueryIntent.EFFECT:
        rel_filter = (parsed.target_relation,) if parsed.target_relation else _EFFECT_RELATIONS
        for rel in rel_filter:
            rows.extend(kg.out_edges(primary, relation=rel))
        res.triples = sorted(_dedup(rows), key=lambda t: -t.score)[:10]

    elif intent == QueryIntent.LOCATION:
        rows = kg.out_edges(primary, relation="located_at")
        res.triples = sorted(rows, key=lambda t: -t.score)[:8]

    elif intent == QueryIntent.CHAPTER:
        rows = kg.out_edges(primary, relation="discussed_in")
        res.triples = sorted(rows, key=lambda t: -t.score)[:8]

    elif intent == QueryIntent.PROPERTY:
        target_rel = parsed.target_relation
        if target_rel:
            rows = kg.out_edges(primary, relation=target_rel)
        if not rows:
            for rel in _PROPERTY_RELATIONS:
                rows.extend(kg.out_edges(primary, relation=rel))
        res.triples = sorted(_dedup(rows), key=lambda t: -t.score)[:12]

    elif intent == QueryIntent.RELATION and secondary:
        direct = kg.related(primary, secondary)
        rows = list(direct)
        if not direct:
            paths = kg.find_paths(primary, secondary, max_hops=3, max_paths=4)
            for path in paths:
                rows.extend(path)
            if paths:
                res.note = f"两实体无直接关系，已展开 {len(paths)} 条多跳路径。"
        res.triples = _dedup(rows)[:15]

    elif intent == QueryIntent.SUMMARY:
        rows = kg.triples_of(primary)
        res.triples = sorted(rows, key=lambda t: -t.score)[:15]

    else:  # NEIGHBOR / UNKNOWN
        rows = kg.triples_of(primary)
        res.triples = sorted(rows, key=lambda t: -t.score)[:12]

    # ---- 原文证据 ----
    if intent == QueryIntent.RELATION and secondary:
        res.evidence_rows = corpus.search_pair(primary, secondary, limit=6)
        if not res.evidence_rows:
            res.evidence_rows = (corpus.search(primary, limit=3) + corpus.search(secondary, limit=3))[:6]
    else:
        res.evidence_rows = corpus.search(primary, limit=6)

    # ---- 辅助显示：实体的其他三元组（与主答案不重复） ----
    if res.subgraph_seed:
        all_of_primary = kg.triples_of(res.subgraph_seed)
        main_keys = {t.key for t in res.triples}
        res.related_triples = [t for t in all_of_primary if t.key not in main_keys][:10]

    return res
