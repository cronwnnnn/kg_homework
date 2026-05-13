"""基于实体类型 / 章节归属的关系抽取。

包含两类：
    1) TypeBasedExtractor：实体 → 类型标签 的 instance_of 关系（主力，贡献 94% TP）。
    2) ChapterMembershipExtractor：实体 → 章节标题 的 discussed_in 关系（供 app_qa 章节问答使用）。

辅助函数：
    collect_cooccurring_entities(ner, sentences, entities_by_type) → 收集
        "句子级同类型对"的实体，喂给 TypeBasedExtractor.set_observed_entities 用。
        原 CooccurrenceTypeExtractor 的"产 has_parameter/made_of 等 triple"已删除，
        但实体收集副作用保留，否则 instance_of 召回会损失约 6%。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .ner import HybridNER
from .preprocess import Sentence
from .schema import Triple


# 用于 collect_cooccurring_entities：满足任一类型对的实体进入 observed
_COOCCUR_TYPE_PAIRS: set[tuple[str, str]] = {
    ("AIRCRAFT", "PARAMETER"),
    ("AIRCRAFT", "PERFORMANCE_METRIC"),
    ("AIRCRAFT", "WING_CONFIGURATION"),
    ("AIRCRAFT", "MATERIAL"),
    ("AIRCRAFT", "STRUCTURAL_COMPONENT"),
    ("STRUCTURAL_COMPONENT", "MATERIAL"),
    ("PERSON", "ORGANIZATION"),
    ("TECHNOLOGY", "AIRCRAFT"),
}
# 加入反向
_COOCCUR_TYPE_PAIRS = _COOCCUR_TYPE_PAIRS | {(b, a) for a, b in _COOCCUR_TYPE_PAIRS}


def collect_cooccurring_entities(
    ner: HybridNER,
    sentences: Iterable[Sentence],
    entities_by_type: dict[str, Sequence[str]],
) -> set[str]:
    """扫描所有句子，把"同句出现且类型对在白名单内"的实体收集起来。

    用途：弥补"只在共现中出现、未被任何关系抽取器命中"的实体，让 TypeBasedExtractor
    能给它们打 instance_of 标签（这是 F1 主力贡献的关键）。
    """
    term_to_type: dict[str, str] = {}
    for type_name, items in entities_by_type.items():
        for w in items:
            w = (w or "").strip()
            if w:
                term_to_type.setdefault(w, type_name)

    observed: set[str] = set()
    for sent in sentences:
        typed = [(m.text, t) for m in ner.extract(sent.text)
                 if (t := term_to_type.get(m.text)) is not None]
        if len(typed) < 2:
            continue
        for i, (a_text, a_type) in enumerate(typed):
            for b_text, b_type in typed[i + 1:]:
                if a_text == b_text:
                    continue
                if (a_type, b_type) in _COOCCUR_TYPE_PAIRS:
                    observed.add(a_text)
                    observed.add(b_text)
    return observed


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
