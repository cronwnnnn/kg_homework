"""混合实体识别：词典匹配（AC 自动机）+ 数值/单位正则 + 人物机构启发式。"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .schema import Mention

try:
    import ahocorasick  # type: ignore
except ImportError:  # 允许在未安装 pyahocorasick 时降级到朴素匹配
    ahocorasick = None


_NUMERIC_PATTERNS = [
    # 带单位的常见数值：3000m / 400km/h / 0.8Mach / 25% / 12t / 9克 / 80000N·m
    re.compile(r"\d+(?:\.\d+)?\s*(?:km/h|m/s|kt|kg|t|g|kw|kW|m|cm|mm|s|ms|km|kn|N·m|N\u00b7m|Nm|N|度|°|°C|%|Mach|马赫|克|吨|公斤|公里|米|秒|赫兹|Hz|分钟|小时|h)"),
    re.compile(r"\d+(?:\.\d+)?[\u00d7x]\d+(?:\.\d+)?\^?\d+"),  # 7×10^5
    re.compile(r"\d+\s*-\s*\d+\s*(?:km/h|m/s|kg|t|m|cm|mm|°|度|%|度/秒|kW|kw|N|N·m|Mach)"),
]

_PURE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_AIRCRAFT_CODE_RE = re.compile(r"\b[A-Z]{1,3}-?\d{1,3}[A-Z]?\b")  # F-111, B-1, MQ-9, X-47B, RQ-4A
_FORMULA_LIKE_RE = re.compile(r"[A-Za-z]_?[A-Za-z0-9αβγδθλπμσφψω]+")


class HybridNER:
    """混合实体识别器。

    支持四类来源：
        - vocab：领域词典（来自 ans.py 的实体库），AC 自动机 O(N) 匹配；
        - numeric：带单位的数值；
        - aircraft_code：型号编码（F-111、MQ-9 等）；
        - chapter_anchors：章节标题中提取的实体也入库。
    """

    def __init__(
        self,
        vocab: Iterable[str],
        enable_numeric: bool = True,
        enable_aircraft_code: bool = True,
    ) -> None:
        self._vocab: list[str] = sorted({v.strip() for v in vocab if v and v.strip()}, key=lambda x: -len(x))
        self._vocab_set = set(self._vocab)
        self.enable_numeric = enable_numeric
        self.enable_aircraft_code = enable_aircraft_code

        self._automaton = None
        if ahocorasick is not None:
            self._automaton = ahocorasick.Automaton()
            for idx, word in enumerate(self._vocab):
                if len(word) >= 2:
                    self._automaton.add_word(word, (idx, word))
            if len(self._vocab) > 0:
                self._automaton.make_automaton()

    @property
    def vocab_set(self) -> set[str]:
        return self._vocab_set

    def add_terms(self, terms: Sequence[str]) -> None:
        new_terms = [t for t in terms if t not in self._vocab_set and len(t) >= 2]
        if not new_terms:
            return
        self._vocab.extend(new_terms)
        self._vocab_set.update(new_terms)
        if ahocorasick is not None:
            self._automaton = ahocorasick.Automaton()
            for idx, word in enumerate(self._vocab):
                self._automaton.add_word(word, (idx, word))
            self._automaton.make_automaton()

    def extract(self, text: str) -> list[Mention]:
        mentions: list[Mention] = []
        mentions.extend(self._extract_vocab(text))
        if self.enable_aircraft_code:
            mentions.extend(self._extract_aircraft_code(text))
        if self.enable_numeric:
            mentions.extend(self._extract_numeric(text))
        return self._dedup_overlap(mentions)

    def _extract_vocab(self, text: str) -> list[Mention]:
        out: list[Mention] = []
        if self._automaton is not None and self._vocab:
            for end_index, (_, word) in self._automaton.iter(text):
                start = end_index - len(word) + 1
                out.append(Mention(text=word, start=start, end=end_index + 1, etype="vocab"))
        else:
            for w in self._vocab:
                if len(w) < 2:
                    continue
                start = text.find(w)
                while start != -1:
                    out.append(Mention(text=w, start=start, end=start + len(w), etype="vocab"))
                    start = text.find(w, start + 1)
        return out

    @staticmethod
    def _extract_aircraft_code(text: str) -> list[Mention]:
        out: list[Mention] = []
        for m in _AIRCRAFT_CODE_RE.finditer(text):
            out.append(Mention(text=m.group(0), start=m.start(), end=m.end(), etype="aircraft_code"))
        return out

    @staticmethod
    def _extract_numeric(text: str) -> list[Mention]:
        out: list[Mention] = []
        for pat in _NUMERIC_PATTERNS:
            for m in pat.finditer(text):
                out.append(Mention(text=m.group(0).strip(), start=m.start(), end=m.end(), etype="numeric"))
        return out

    @staticmethod
    def _dedup_overlap(mentions: list[Mention]) -> list[Mention]:
        """去除重叠：长 mention 优先，同范围去重。"""
        if not mentions:
            return []
        mentions = sorted(mentions, key=lambda m: (m.start, -(m.end - m.start)))
        kept: list[Mention] = []
        for m in mentions:
            if kept and m.start < kept[-1].end and (m.end - m.start) <= (kept[-1].end - kept[-1].start):
                continue
            if kept and m.start == kept[-1].start and m.end == kept[-1].end:
                continue
            kept.append(m)
        kept.sort(key=lambda x: x.start)
        return kept
