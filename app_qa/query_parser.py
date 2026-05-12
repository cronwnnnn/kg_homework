"""自然语言问题解析：识别问题类型 + 抽取问题中的实体。

问题类型 (QueryIntent)：
    DEFINITION      X 是什么 / 什么是 X
    PROPERTY        X 的 Y / X 有哪些 Y / X 有什么 Y
    VALUE           X 是多少 / X 的值
    CAUSE           什么导致 X / 什么减少了 X / 什么影响 X
    EFFECT          X 会导致什么 / X 影响什么 / X 减少什么
    LIST_INSTANCE   有哪些 X / 列举 X
    LOCATION        X 在哪 / X 位于哪
    CHAPTER         X 在哪章讨论 / X 出现在哪
    RELATION        X 和 Y / X 与 Y 的关系 / X 跟 Y
    NEIGHBOR        与 X 相关的 / X 的邻居 / X 周围
    SUMMARY         总结 X / X 概述
    UNKNOWN         默认兜底（按 NEIGHBOR 处理）
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from extractors.ner import HybridNER  # noqa: E402


class QueryIntent(Enum):
    DEFINITION = "definition"
    PROPERTY = "property"
    VALUE = "value"
    CAUSE = "cause"
    EFFECT = "effect"
    LIST_INSTANCE = "list_instance"
    LOCATION = "location"
    CHAPTER = "chapter"
    RELATION = "relation"
    NEIGHBOR = "neighbor"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    raw: str
    intent: QueryIntent
    entities: list[str]
    target_relation: str | None = None  # 例如 PROPERTY 时希望查的关系名

    @property
    def primary(self) -> str:
        return self.entities[0] if self.entities else ""

    @property
    def secondary(self) -> str:
        return self.entities[1] if len(self.entities) >= 2 else ""


_PATTERNS: list[tuple[QueryIntent, re.Pattern[str], str | None]] = [
    # (intent, pattern, optional target_relation hint)
    (QueryIntent.DEFINITION, re.compile(r"^(.+?)是什么(\?|？)?$"), "instance_of"),
    (QueryIntent.DEFINITION, re.compile(r"^什么是(.+?)(\?|？)?$"), "instance_of"),
    (QueryIntent.DEFINITION, re.compile(r"(?:解释|定义)(.+?)(?:[\?？]|$)"), "instance_of"),
    (QueryIntent.LIST_INSTANCE, re.compile(r"(?:有哪些|哪些是|列举).*?(.+?)(?:[\?？]|$)"), "instance_of"),
    (QueryIntent.VALUE, re.compile(r"(.+?)(?:是)?多少(\?|？)?$"), "has_value"),
    (QueryIntent.VALUE, re.compile(r"(.+?)的值(?:是多少)?(\?|？)?$"), "has_value"),
    (QueryIntent.CAUSE, re.compile(r"(?:什么|哪些).*?(?:导致|引起|造成)(.+?)(?:[\?？]|$)"), "causes"),
    (QueryIntent.CAUSE, re.compile(r"(?:什么|哪些).*?(?:减少|降低)(.+?)(?:[\?？]|$)"), "reduces"),
    (QueryIntent.CAUSE, re.compile(r"(?:什么|哪些).*?(?:提高|增强|提升|改善)(.+?)(?:[\?？]|$)"), "improves"),
    (QueryIntent.CAUSE, re.compile(r"(?:什么|哪些).*?影响(.+?)(?:[\?？]|$)"), "affects"),
    (QueryIntent.EFFECT, re.compile(r"(.+?)(?:会|能)?(?:导致|引起|造成)什么"), "causes"),
    (QueryIntent.EFFECT, re.compile(r"(.+?)(?:会|能)?(?:减少|降低)什么"), "reduces"),
    (QueryIntent.EFFECT, re.compile(r"(.+?)(?:会|能)?(?:提高|增强|提升|改善)什么"), "improves"),
    (QueryIntent.EFFECT, re.compile(r"(.+?)(?:会|能)?影响什么"), "affects"),
    (QueryIntent.LOCATION, re.compile(r"(.+?)(?:在|位于)哪"), "located_at"),
    (QueryIntent.CHAPTER, re.compile(r"(.+?)(?:在哪章|出现在哪|哪章讨论)"), "discussed_in"),
    (QueryIntent.PROPERTY, re.compile(r"(.+?)的(.+?)是(?:什么|多少|哪些)"), None),
    (QueryIntent.PROPERTY, re.compile(r"(.+?)有(?:什么|哪些)(.+?)(?:[\?？]|$)"), None),
    (QueryIntent.PROPERTY, re.compile(r"(.+?)由什么组成"), "has_part"),
    (QueryIntent.PROPERTY, re.compile(r"(.+?)包(?:括|含)(?:什么|哪些)"), "has_part"),
    (QueryIntent.RELATION, re.compile(r"(.+?)(?:和|与|跟)(.+?)(?:的关系|的联系|有什么关系)"), None),
    (QueryIntent.NEIGHBOR, re.compile(r"(?:与|和|跟)(.+?)相关(?:的)?"), None),
    (QueryIntent.NEIGHBOR, re.compile(r"(.+?)(?:的)?(?:邻居|相邻|周围)"), None),
    (QueryIntent.SUMMARY, re.compile(r"^(?:总结|概述|介绍)(.+?)(?:[\?？]|$)"), None),
]


_SUFFIX_NOISE = "?？.。、!！,，;；:："
_PREFIX_NOISE = " 请问帮我"


def _clean(s: str) -> str:
    s = (s or "").strip()
    while s and s[-1] in _SUFFIX_NOISE:
        s = s[:-1]
    while s and s[0] in _PREFIX_NOISE:
        s = s[1:]
    return s.strip()


class QueryParser:
    def __init__(self, vocab: Iterable[str]) -> None:
        self.ner = HybridNER(vocab=vocab, enable_numeric=False, enable_aircraft_code=True)
        self.vocab_set = self.ner.vocab_set

    def parse(self, question: str) -> ParsedQuery:
        q = _clean(question)

        # 1) 先匹配模板
        intent = QueryIntent.UNKNOWN
        target_rel: str | None = None
        captured: list[str] = []
        for it, pat, rel_hint in _PATTERNS:
            m = pat.search(q)
            if m:
                intent = it
                target_rel = rel_hint
                captured = [_clean(g) for g in m.groups() if g and g not in "?？"]
                break

        # 2) 再做 NER 抽取实体（包括模板捕获组）
        mentions = self.ner.extract(q)
        ents: list[str] = []
        seen: set[str] = set()
        for m in sorted(mentions, key=lambda mm: mm.start):
            if m.text and m.text not in seen:
                seen.add(m.text)
                ents.append(m.text)

        # 3) 若模板捕获组里有词典实体，提到最前
        cap_ents: list[str] = []
        for c in captured:
            for w in sorted(self.vocab_set, key=lambda x: -len(x)):
                if len(w) >= 2 and w in c and w not in cap_ents:
                    cap_ents.append(w)
                    break
        if cap_ents:
            for e in reversed(cap_ents):
                if e in ents:
                    ents.remove(e)
                ents.insert(0, e)

        return ParsedQuery(raw=q, intent=intent, entities=ents, target_relation=target_rel)
