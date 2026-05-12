"""论文原文句子索引：把 aftcln.txt 切分为句子，并建立 实体 → 句子位置 倒排索引。

提供：
    Corpus.sentences            list[SentenceRecord]
    Corpus.search(entity)       返回包含该实体的所有句子（按章节排序）
    Corpus.search_any(*ents)    返回包含任意一个实体的句子
    Corpus.search_all(*ents)    返回同时包含所有实体的句子
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from extractors.preprocess import TextPreprocessor  # noqa: E402


@dataclass(frozen=True)
class SentenceRecord:
    sid: int
    text: str
    chapter: str
    section: str
    paragraph_id: int


class Corpus:
    def __init__(self, text_path: str) -> None:
        if not os.path.isfile(text_path):
            raise FileNotFoundError(f"找不到论文文件: {text_path}")
        raw = TextPreprocessor.load_text(text_path)
        pre = TextPreprocessor()
        _, sentences = pre.process(raw)
        self.sentences: list[SentenceRecord] = [
            SentenceRecord(
                sid=s.sentence_id,
                text=s.text,
                chapter=s.chapter,
                section=s.section,
                paragraph_id=s.paragraph_id,
            )
            for s in sentences
        ]
        # 朴素倒排：实体子串 → 句子下标（按需要查询时再线性扫描，避免预构建巨大 dict）
        # 但保留一个排序结构供 search() 用 in 操作
        self._cached_index: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.sentences)

    def search(self, entity: str, limit: int = 20) -> list[SentenceRecord]:
        if not entity:
            return []
        if entity in self._cached_index:
            idxs = self._cached_index[entity]
        else:
            idxs = [i for i, s in enumerate(self.sentences) if entity in s.text]
            self._cached_index[entity] = idxs
        return [self.sentences[i] for i in idxs[:limit]]

    def search_any(self, *entities: str, limit: int = 20) -> list[SentenceRecord]:
        seen: set[int] = set()
        out: list[SentenceRecord] = []
        for e in entities:
            for i, s in enumerate(self.sentences):
                if i in seen:
                    continue
                if e and e in s.text:
                    seen.add(i)
                    out.append(s)
                    if len(out) >= limit:
                        return out
        return out

    def search_all(self, *entities: str, limit: int = 20) -> list[SentenceRecord]:
        ents = [e for e in entities if e]
        if not ents:
            return []
        out: list[SentenceRecord] = []
        for s in self.sentences:
            if all(e in s.text for e in ents):
                out.append(s)
                if len(out) >= limit:
                    break
        return out

    def search_pair(self, a: str, b: str, limit: int = 10) -> list[SentenceRecord]:
        """同时包含 a 和 b 的句子，优先短句。"""
        hits = self.search_all(a, b, limit=limit * 2)
        hits.sort(key=lambda s: len(s.text))
        return hits[:limit]
