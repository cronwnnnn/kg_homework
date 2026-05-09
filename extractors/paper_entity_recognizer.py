"""论文内实体挖掘：在领域词表之外，用语义/词性信息补充「文中真实出现」的实体串。

与 HybridNER 的关系：
    - HybridNER 仍以领域词表 + 数值 + 型号码为主力；
    - 本模块从 spaCy 的命名实体、名词短语、名/专名连续串中收集候选，
      经停用词与频次过滤后，通过 HybridNER.add_terms 并入 AC 自动机，
      使后续触发词、共现、SVO 等能在「词表未覆盖但文中确有」的串上成边。

注意：会引入一定噪声，可用 ``paper_entity_min_doc_freq`` 与关闭开关控制。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .preprocess import Sentence


# 高频虚词 / 论文章节用语，不作为实体入库
_ENTITY_STOP = frozenset(
    {
        "的", "了", "和", "与", "或", "及", "等", "中", "为", "在", "有", "对", "将", "被", "由", "从",
        "可以", "能够", "需要", "进行", "具有", "采用", "通过", "以及", "因此", "其中", "此时",
        "本文", "本章", "本节", "上述", "如下", "所示", "情况", "问题", "方面", "过程", "结果",
        "部分", "方式", "方法", "研究", "分析", "讨论", "结论", "内容", "结构", "系统", "模型",
        "数据", "结果", "水平", "条件", "状态", "关系", "影响", "变化", "增加", "减小", "提高",
        "降低", "较大", "较小", "不同", "相同", "主要", "重要", "一般", "通常", "可能", "如果",
    }
)

_MAX_PHRASE_LEN = 14
_MIN_LEN = 2
_MAX_LEN = 28


def _clean_span(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^[\s\"'「『（(\[【]+", "", s)
    s = re.sub(r"[\s\"'」』）)\]】、，。；：！？]+$", "", s)
    return s


def _ok_span(s: str, vocab: set[str]) -> bool:
    if not s or len(s) < _MIN_LEN or len(s) > _MAX_LEN:
        return False
    if s in _ENTITY_STOP or s in vocab:
        return False
    if re.fullmatch(r"[\d\s.%×xX^\-]+", s):
        return False
    # 至少含一个汉字，减少纯英文碎片
    if not re.search(r"[\u4e00-\u9fff]", s):
        return len(s) >= 4
    return True


class PaperEntityRecognizer:
    """基于 spaCy 的论文实体挖掘器（不替代词典，只做补充）。"""

    def __init__(self, spacy_model: str = "zh_core_web_sm") -> None:
        self.spacy_model = spacy_model
        self._nlp = None

    def _load(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
        except Exception:  # noqa: BLE001
            self._nlp = False  # type: ignore[assignment]
        return self._nlp

    def collect_from_sentences(
        self,
        sentences: Iterable["Sentence"],
        vocab_boost: set[str],
        min_doc_freq: int = 1,
    ) -> list[str]:
        """返回按频次排序后的新实体串列表（已排除 vocab_boost 内已有词）。"""
        nlp = self._load()
        if not nlp or nlp is False:
            return []

        raw_counter: Counter[str] = Counter()
        for sent in sentences:
            text = (sent.text or "").strip()
            if len(text) < _MIN_LEN:
                continue
            doc = nlp(text)
            for ent in doc.ents:
                span = _clean_span(ent.text)
                if _ok_span(span, vocab_boost):
                    raw_counter[span] += 1
            try:
                for nc in doc.noun_chunks:
                    span = _clean_span(nc.text)
                    if _ok_span(span, vocab_boost):
                        raw_counter[span] += 1
            except Exception:  # noqa: BLE001
                pass
            self._collect_noun_runs(doc, raw_counter, vocab_boost)

        out: list[str] = []
        for span, c in raw_counter.most_common():
            if span in vocab_boost:
                continue
            if c < min_doc_freq:
                continue
            out.append(span)
        # 长串优先：避免短串覆盖长串的 add_terms 顺序问题交给 HybridNER 排序
        out.sort(key=lambda x: (-len(x), x))
        return out

    def _collect_noun_runs(self, doc, raw_counter: Counter[str], vocab_boost: set[str]) -> None:
        """合并连续的名词/专名片段（中文 spaCy 常切成单字，需拼成短语）。"""
        toks = [t for t in doc if not t.is_space]
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.pos_ not in ("NOUN", "PROPN"):
                i += 1
                continue
            parts: list[str] = [t.text]
            j = i + 1
            while j < len(toks) and toks[j].pos_ in ("NOUN", "PROPN"):
                cand = "".join(parts) + toks[j].text
                if len(cand) > _MAX_PHRASE_LEN:
                    break
                parts.append(toks[j].text)
                j += 1
            span = _clean_span("".join(parts))
            if _ok_span(span, vocab_boost):
                raw_counter[span] += 1
            i = j if j > i else i + 1
