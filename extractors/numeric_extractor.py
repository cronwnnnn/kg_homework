"""数值/参数关系抽取：识别"X 为 N单位"等参数赋值句式。"""

from __future__ import annotations

import re
from typing import Iterable

from .ner import HybridNER
from .preprocess import Sentence
from .schema import Triple


_UNITS = (
    r"km/h|m/s|kt|kg|t|g|kw|kW|mw|MW|cm|mm|m|km|s|ms|min|h|N·m|N\u00b7m|Nm|N|"
    r"度|°|°C|%|Mach|马赫|克|吨|公斤|公里|米|秒|分钟|小时|赫兹|Hz|Pa|kPa|MPa"
)
_VALUE_RE = re.compile(rf"(?P<num>\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?\d+)?)\s*(?P<unit>{_UNITS})?")

_PATTERNS = [
    re.compile(rf"(?P<head>[\u4e00-\u9fa5A-Za-z·\-]+?)(?:为|是|约为|约等于|大约为|达到了?|可达|不超过|超过)(?P<num>\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?\d+)?)\s*(?P<unit>{_UNITS})"),
    re.compile(rf"(?P<head>[\u4e00-\u9fa5A-Za-z·\-]+?)在(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>{_UNITS})(?:左右|附近|以上|以下)"),
    re.compile(rf"(?P<head>[\u4e00-\u9fa5A-Za-z·\-]+?)(?:大于|高于|不低于|至少)(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>{_UNITS})"),
    re.compile(rf"(?P<head>[\u4e00-\u9fa5A-Za-z·\-]+?)(?:小于|低于|不超过|至多)(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>{_UNITS})"),
]


class NumericExtractor:
    """提取参数 → 数值 的关系。"""

    REL_HAS_VALUE = "has_value"
    REL_GT = "greater_than_value"
    REL_LT = "less_than_value"

    def __init__(self, ner: HybridNER) -> None:
        self.ner = ner
        self.vocab_set = ner.vocab_set

    def extract_from_sentence(self, sent: Sentence) -> list[Triple]:
        text = sent.text
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        for idx, pat in enumerate(_PATTERNS):
            rel = self.REL_HAS_VALUE
            if idx == 2:
                rel = self.REL_GT
            elif idx == 3:
                rel = self.REL_LT
            for m in pat.finditer(text):
                head_raw = m.group("head").strip()
                num = m.group("num").replace(" ", "")
                unit = m.group("unit") or ""
                value = f"{num}{unit}"
                heads = self._resolve(head_raw)
                for h in heads:
                    if not h:
                        continue
                    key = (h, rel, value)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(
                        Triple(
                            head=h,
                            relation=rel,
                            tail=value,
                            trigger="numeric",
                            source="numeric",
                            score=0.78,
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

    def _resolve(self, fragment: str) -> list[str]:
        fragment = fragment.strip()
        if not fragment:
            return []
        if fragment in self.vocab_set:
            return [fragment]
        candidates = sorted(self.vocab_set, key=lambda x: -len(x))
        for w in candidates:
            if len(w) < 2:
                continue
            if w in fragment:
                return [w]
        return []
