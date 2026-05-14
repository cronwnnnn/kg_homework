"""关系模板抽取：高精度的中文固定句式识别。

"""

from __future__ import annotations

import re
from typing import Iterable

from .ner import HybridNER
from .preprocess import Sentence
from .schema import Triple


_LIST_SEP = re.compile(r"[、，,;；]")

# (trigger_pattern, relation, name, allow_list_tail)
_PATTERNS: list[tuple[re.Pattern[str], str, str, bool]] = [
    (re.compile(r"由(.{0,8}?)驱动"), "driven_by", "由X驱动", False),
    (re.compile(r"(?:包括|包含|含有|分为)"), "has_part", "包括", True),
    (re.compile(r"由(.{0,12}?)组成"), "has_part", "由X组成", True),
    (re.compile(r"(?:用于|适用于|应用于)"), "used_for", "用于", False),
    (re.compile(r"起源于"), "originates_from", "起源于", False),
    (re.compile(r"(?:研制|生产|制造)出?了?"), "manufactures", "研制/制造", False),
    (re.compile(r"(?:研发|开发)出?了?"), "develops", "研发/开发", False),
    (re.compile(r"是一种?"), "is_a", "是一种", False),
    (re.compile(r"位于"), "located_at", "位于", False),
    (re.compile(r"安装(?:在|于)"), "located_at", "安装于", False),
    # 连接/对接：要求显式 "与/和 …… 连接" 才认，不再吃光所有"连接"动词
    (re.compile(r"(?:转变为|变体为|切换为)"), "transforms_to", "转变为", False),
]


# 双实体型模式：要求 head … 中介词 … tail … 触发词 形式
# (full_pattern, relation, name)
_BIENTITY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"与(.{0,12}?)(?:连接|对接|铰接|固连)"), "connected_to", "与X连接"),
    (re.compile(r"和(.{0,12}?)(?:连接|对接|铰接|固连)"), "connected_to", "和X连接"),
]


_HEAD_LOOKBACK = 12
_TAIL_LOOKAHEAD = 16


class PatternExtractor:
    """基于正则模板的高精度抽取。"""

    def __init__(self, ner: HybridNER) -> None:
        self.ner = ner
        self.vocab_set = ner.vocab_set
        self._sorted_vocab = sorted(self.vocab_set, key=lambda x: -len(x))

    def extract_from_sentence(self, sent: Sentence) -> list[Triple]:
        text = sent.text
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for pat, rel, name, allow_list_tail in _PATTERNS:
            for m in pat.finditer(text):
                tstart, tend = m.start(), m.end()
                head_window = text[max(0, tstart - _HEAD_LOOKBACK): tstart]
                tail_window = text[tend: min(len(text), tend + _TAIL_LOOKAHEAD)]
                head = self._right_most_in(head_window)
                tails: list[str] = []
                if allow_list_tail:
                    chunks = [c for c in _LIST_SEP.split(tail_window) if c.strip()]
                    for c in chunks[:4]:
                        ent = self._left_most_in(c)
                        if ent:
                            tails.append(ent)
                else:
                    ent = self._left_most_in(tail_window)
                    if ent:
                        tails.append(ent)
                if not head or not tails:
                    continue
                for t in tails:
                    if t == head:
                        continue
                    key = (head, rel, t)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(
                        Triple(
                            head=head,
                            relation=rel,
                            tail=t,
                            trigger=name,
                            source="pattern",
                            score=0.82,
                            chapter=sent.chapter,
                            sentence=text,
                        )
                    )

        # 双实体型模式（X 与 Y 连接）
        for pat, rel, name in _BIENTITY_PATTERNS:
            for m in pat.finditer(text):
                tstart = m.start()  # "与" 的位置
                tend = m.end()      # 触发词结束位置
                # head: "与" 左侧最近的词典实体
                head_window = text[max(0, tstart - _HEAD_LOOKBACK): tstart]
                head = self._right_most_in(head_window)
                # tail: 在 group(1) 中找词典实体
                middle = m.group(1)
                tail = self._left_most_in(middle)
                if not head or not tail or head == tail:
                    continue
                key = (head, rel, tail)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    Triple(
                        head=head,
                        relation=rel,
                        tail=tail,
                        trigger=name,
                        source="pattern",
                        score=0.85,
                        chapter=sent.chapter,
                        sentence=text,
                    )
                )
        return triples

    def extract_from_sentences(self, sentences: Iterable[Sentence]) -> list[Triple]:
        out: list[Triple] = []
        for s in sentences:
            out.extend(self.extract_from_sentence(s))
        return out

    def _right_most_in(self, fragment: str) -> str | None:
        """取片段中最靠右且最长的词典实体（更可能是触发词左侧的主语）。"""
        if not fragment:
            return None
        best: tuple[int, int, str] | None = None  # (right_pos, length, word)
        for w in self._sorted_vocab:
            if len(w) < 2 or w not in fragment:
                continue
            pos = fragment.rfind(w)
            cand = (pos + len(w), len(w), w)
            if best is None or cand > best:
                best = cand
        return best[2] if best else None

    def _left_most_in(self, fragment: str) -> str | None:
        """取片段中最靠左且最长的词典实体（更可能是触发词右侧的宾语）。"""
        if not fragment:
            return None
        best: tuple[int, int, str] | None = None  # (-left_pos, length, word)
        for w in self._sorted_vocab:
            if len(w) < 2 or w not in fragment:
                continue
            pos = fragment.find(w)
            cand = (-pos, len(w), w)
            if best is None or cand > best:
                best = cand
        return best[2] if best else None
