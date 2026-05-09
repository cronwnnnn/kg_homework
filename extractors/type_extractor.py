"""基于实体类型的关系抽取。

包含三类：
    1) TypeBasedExtractor：实体 → 类型标签 的 instance_of 关系。
    2) CooccurrenceTypeExtractor：句子级共现 + 跨类型规则的关系。
    3) ChapterMembershipExtractor：实体 → 章节标题 的 discussed_in 关系。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .ner import HybridNER
from .preprocess import Sentence
from .schema import Triple


_TYPE_LABEL_MAP: dict[str, str] = {
    "AIRCRAFT": "飞行器",
    "WING_CONFIGURATION": "机翼构型",
    "PARAMETER": "设计参数",
    "AERODYNAMIC_CONCEPT": "气动概念",
    "STRUCTURAL_COMPONENT": "结构部件",
    "CONTROL_METHOD": "控制方法",
    "PERFORMANCE_METRIC": "性能指标",
    "ORGANIZATION": "组织机构",
    "PERSON": "人物",
    "TECHNOLOGY": "技术",
    "MATERIAL": "材料",
    "FLIGHT_PHASE": "飞行阶段",
    "EQUATION": "公式",
    "CONCEPT": "概念",
}

# 跨类型规则：(left_type, right_type, relation)
# 这些关系仅靠 "类型 + 同句共现" 推得，准确率本身就低于触发词抽取。
# 因此：
#   - 选择"语义弱、容错度高"的关系名（has_parameter / has_performance / co_occurs_with）；
#   - 涉及"研发/制造/控制"等强语义动作的，留给 trigger / pattern 抽取，cooccur 不做。
_CROSS_TYPE_RULES: list[tuple[str, str, str]] = [
    ("AIRCRAFT", "PARAMETER", "has_parameter"),
    ("AIRCRAFT", "PERFORMANCE_METRIC", "has_performance"),
    ("AIRCRAFT", "WING_CONFIGURATION", "has_configuration"),
    ("AIRCRAFT", "MATERIAL", "uses_material"),
    ("AIRCRAFT", "STRUCTURAL_COMPONENT", "has_part"),
    ("STRUCTURAL_COMPONENT", "MATERIAL", "made_of"),
    ("PERSON", "ORGANIZATION", "affiliated_with"),
    ("TECHNOLOGY", "AIRCRAFT", "applied_to"),
]


class TypeBasedExtractor:
    """实体 → 类型标签 的 instance_of 关系生成器。"""

    def __init__(self, entities_by_type: dict[str, Sequence[str]]) -> None:
        self.entities_by_type: dict[str, list[str]] = {k: list(v) for k, v in entities_by_type.items()}
        self.observed: set[str] = set()

    def set_observed_entities(self, observed: Iterable[str]) -> None:
        self.observed = {(e or "").strip() for e in observed if e}

    def extract(self) -> list[Triple]:
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for type_name, items in self.entities_by_type.items():
            type_label = _TYPE_LABEL_MAP.get(type_name)
            if not type_label:
                continue
            for e in items:
                e = (e or "").strip()
                if not e or e not in self.observed:
                    continue
                key = (e, "instance_of", type_label)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        head=e,
                        relation="instance_of",
                        tail=type_label,
                        trigger="type_taxonomy",
                        source="type",
                        score=0.92,
                    )
                )
        return triples


class CooccurrenceTypeExtractor:
    """句子级共现 + 跨类型规则。

    工作原理：
        1) 先用 NER 在每个句子中识别词典实体；
        2) 在同一个句子里，对所有不同类型对的实体两两组合；
        3) 若类型对命中 _CROSS_TYPE_RULES，则产生一条三元组。
    """

    def __init__(
        self,
        ner: HybridNER,
        entities_by_type: dict[str, Sequence[str]],
        max_pairs_per_sentence: int = 8,
    ) -> None:
        self.ner = ner
        self.max_pairs_per_sentence = max_pairs_per_sentence
        # 反向索引：词 → 类型
        self._term_to_type: dict[str, str] = {}
        for type_name, items in entities_by_type.items():
            for w in items:
                w = (w or "").strip()
                if not w:
                    continue
                self._term_to_type.setdefault(w, type_name)

        # 规则集查表：(left_type, right_type) → relation
        self._rule_map: dict[tuple[str, str], str] = {(s, t): r for s, t, r in _CROSS_TYPE_RULES}
        # 也允许反向
        self._reverse_rule_map: dict[tuple[str, str], str] = {(t, s): r for s, t, r in _CROSS_TYPE_RULES}

    def extract_from_sentence(self, sent: Sentence) -> list[Triple]:
        mentions = self.ner.extract(sent.text)
        if len(mentions) < 2:
            return []
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        # 按出现顺序选实体并分配类型
        typed: list[tuple[str, str]] = []
        for m in mentions:
            t = self._term_to_type.get(m.text)
            if t is None:
                continue
            typed.append((m.text, t))
        # 限制每句关系数量（抑制噪声）
        budget = self.max_pairs_per_sentence
        for i in range(len(typed)):
            for j in range(len(typed)):
                if i == j or budget <= 0:
                    continue
                a_text, a_type = typed[i]
                b_text, b_type = typed[j]
                if a_text == b_text:
                    continue
                rel = self._rule_map.get((a_type, b_type))
                if rel is None:
                    continue
                key = (a_text, rel, b_text)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        head=a_text,
                        relation=rel,
                        tail=b_text,
                        trigger=f"cooccur:{a_type}->{b_type}",
                        source="cooccur",
                        score=0.50,
                        chapter=sent.chapter,
                        sentence=sent.text,
                    )
                )
                budget -= 1
        return triples

    def extract_from_sentences(self, sentences: Iterable[Sentence]) -> list[Triple]:
        out: list[Triple] = []
        for s in sentences:
            out.extend(self.extract_from_sentence(s))
        return out


class ChapterMembershipExtractor:
    """章节归属：在某章节出现频次 ≥ N 的实体 → discussed_in → 章节标题。

    限制：
        1. 仅对在该章节出现 ≥ min_occur 次的实体建立关系；
        2. 仅建立到"非空章节标题"；
        3. 用 SECONDARY_TYPES 限制实体类型，避免数值类参与。
    """

    SECONDARY_TYPES: set[str] = {
        "AIRCRAFT", "WING_CONFIGURATION", "TECHNOLOGY", "STRUCTURAL_COMPONENT",
        "CONTROL_METHOD", "ORGANIZATION", "PERSON", "EQUATION", "MATERIAL",
        "PARAMETER", "AERODYNAMIC_CONCEPT",
    }

    def __init__(
        self,
        ner: HybridNER,
        entities_by_type: dict[str, Sequence[str]],
        min_occur: int = 2,
    ) -> None:
        self.ner = ner
        self.min_occur = min_occur
        self._term_to_type: dict[str, str] = {}
        for type_name, items in entities_by_type.items():
            if type_name not in self.SECONDARY_TYPES:
                continue
            for w in items:
                w = (w or "").strip()
                if not w:
                    continue
                self._term_to_type.setdefault(w, type_name)

    def extract_from_sentences(self, sentences: Iterable[Sentence]) -> list[Triple]:
        chapter_counter: dict[str, Counter[str]] = defaultdict(Counter)
        chapter_sample_sentence: dict[str, dict[str, str]] = defaultdict(dict)
        for s in sentences:
            chapter = s.chapter or ""
            if not chapter:
                continue
            mentions = self.ner.extract(s.text)
            for m in mentions:
                if m.text in self._term_to_type:
                    chapter_counter[chapter][m.text] += 1
                    chapter_sample_sentence[chapter].setdefault(m.text, s.text)

        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for chapter, counter in chapter_counter.items():
            for ent, n in counter.most_common():
                if n < self.min_occur:
                    continue
                key = (ent, "discussed_in", chapter)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        head=ent,
                        relation="discussed_in",
                        tail=chapter,
                        trigger="chapter_membership",
                        source="chapter",
                        score=0.55 + min(0.25, n * 0.02),
                        chapter=chapter,
                        sentence=chapter_sample_sentence[chapter].get(ent, ""),
                    )
                )
        return triples
