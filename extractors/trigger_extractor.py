"""核心算法：基于触发词 + 共现窗口的关系抽取。

中文 IE 实践中最实用的强 baseline：
    1. 一个句子中识别出的实体两两组合（按位置升序）；
    2. 取实体对中间窗口的文本，匹配触发词；
    3. 触发词命中即映射到统一关系本体；
    4. 根据距离、触发词长度、边界条件给出置信度分数。

相比单纯的 SVO 依存抽取，该方法对中文长句的鲁棒性显著更高，
能从一句话中召回多条三元组。
"""

from __future__ import annotations

from typing import Iterable

from .ner import HybridNER
from .preprocess import Sentence
from .relation_normalizer import RelationNormalizer
from .schema import Mention, RelationOntology, Triple


_NEG_HINTS = ("不", "未", "无", "非", "并未", "没有", "免", "勿")


class TriggerCooccurrenceExtractor:
    """实体对中间触发词扫描器。"""

    def __init__(
        self,
        ner: HybridNER,
        normalizer: RelationNormalizer | None = None,
        max_window: int = 30,
        min_score: float = 0.38,
    ) -> None:
        self.ner = ner
        self.normalizer = normalizer or RelationNormalizer()
        self.max_window = max_window
        self.min_score = min_score
        self._triggers: list[tuple[str, str]] = list(RelationOntology.all_triggers())
        self._weak_triggers = RelationOntology.WEAK_TRIGGERS
        self._relation_constraints = RelationOntology.RELATION_TYPE_CONSTRAINTS

    # ---------- 主接口 ----------

    def extract_from_sentence(self, sent: Sentence) -> list[Triple]:
        mentions = self.ner.extract(sent.text)
        return self._pairwise_extract(sent, mentions)

    def extract_from_sentences(self, sentences: Iterable[Sentence]) -> list[Triple]:
        out: list[Triple] = []
        for s in sentences:
            out.extend(self.extract_from_sentence(s))
        return out

    # ---------- 内部 ----------

    def _pairwise_extract(self, sent: Sentence, mentions: list[Mention]) -> list[Triple]:
        if len(mentions) < 2:
            return []
        text = sent.text
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        mentions = sorted(mentions, key=lambda m: m.start)
        n = len(mentions)
        # 仅对相邻 K 个实体两两组合，避免远距离误关联
        K_NEIGHBOR = 6
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + K_NEIGHBOR)):
                a, b = mentions[i], mentions[j]
                if a.text == b.text:
                    continue
                if a.end > b.start:
                    continue
                gap = text[a.end:b.start]
                if not gap:
                    continue
                if len(gap) > self.max_window:
                    continue
                # 中间不允许跨过句末标点（保险：split_sentences 已切，但中文论文长句仍可能含）
                if any(c in "。！？；" for c in gap):
                    continue
                # 中间出现第三个实体：仅当窗口较短时跳过；窗口较长时保留（容许"X 触发词 Y, Z"的列举）
                third_inside = any(
                    m.start >= a.end and m.end <= b.start and m.text not in (a.text, b.text)
                    for m in mentions
                )
                if third_inside and len(gap) <= 12:
                    continue
                hits = self._match_triggers(gap)
                if not hits:
                    continue
                left_ctx = text[max(0, a.start - 6): a.start]
                right_ctx = text[b.end: min(len(text), b.end + 6)]
                negated = any(neg in gap or neg in left_ctx or neg in right_ctx for neg in _NEG_HINTS)
                for trigger, rel, position in hits:
                    head, tail = a.text, b.text
                    head_etype, tail_etype = a.etype, b.etype
                    # 被动语态：触发词左侧紧邻"由"、"被"、"通过"则反向
                    pre_trigger = gap[:position]
                    if pre_trigger.strip().endswith(("由", "被")) and rel in ("develops", "manufactures"):
                        head, tail = tail, head
                        head_etype, tail_etype = b.etype, a.etype
                    if not self._check_constraints(rel, head_etype, tail_etype):
                        continue
                    score = self._score(gap, trigger, position, a, b, negated)
                    if score < self.min_score:
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
                            trigger=trigger,
                            source="trigger",
                            score=round(score, 3),
                            chapter=sent.chapter,
                            sentence=text,
                        )
                    )
        return triples

    def _check_constraints(self, rel: str, head_etype: str, tail_etype: str) -> bool:
        cstr = self._relation_constraints.get(rel)
        if not cstr:
            return True
        head_forbid = cstr.get("head_forbid", set())
        tail_forbid = cstr.get("tail_forbid", set())
        if head_etype in head_forbid or tail_etype in tail_forbid:
            return False
        return True

    def _match_triggers(self, gap: str) -> list[tuple[str, str, int]]:
        """返回 [(trigger, relation, position_in_gap)]，长触发词优先且不重叠。"""
        hits: list[tuple[str, str, int]] = []
        consumed = [False] * len(gap)
        for trig, rel in self._triggers:
            if trig.startswith("由…") or trig.endswith("…驱动") or "…" in trig:
                continue
            start = gap.find(trig)
            while start != -1:
                if any(consumed[start: start + len(trig)]):
                    start = gap.find(trig, start + 1)
                    continue
                hits.append((trig, rel, start))
                for k in range(start, start + len(trig)):
                    consumed[k] = True
                start = gap.find(trig, start + len(trig))
        return hits

    def _score(
        self,
        gap: str,
        trigger: str,
        position: int,
        head: Mention,
        tail: Mention,
        negated: bool,
    ) -> float:
        # 基础：触发词长度越长越可信
        base = 0.40 + min(len(trigger), 4) * 0.08
        # 距离惩罚：触发词在中间最佳，过远扣分
        gap_extra = max(0, len(gap) - len(trigger))
        base -= 0.30 * min(gap_extra / max(self.max_window, 1), 1.0)
        # 头尾实体长度奖励
        if len(head.text) >= 3:
            base += 0.05
        if len(tail.text) >= 3:
            base += 0.05
        # 否定情境扣分
        if negated:
            base -= 0.30
        # 触发词命中位置：评估"是否紧贴头/尾实体"
        gap_clean = gap.strip()
        prefix = gap[:position]
        suffix = gap[position + len(trigger):]
        prefix_extra = len(prefix.strip("，,的、和与及或而又也"))
        suffix_extra = len(suffix.strip("，,的、和与及或而又也"))
        if prefix_extra == 0 and suffix_extra == 0:
            base += 0.20  # "X 触发词 Y"
        elif prefix_extra <= 2 and suffix_extra <= 2:
            base += 0.10
        elif prefix_extra >= 8 or suffix_extra >= 8:
            base -= 0.15
        # 弱触发词：必须真正"紧贴"实体；否则视为噪声
        if trigger in self._weak_triggers:
            if prefix_extra > 2 or suffix_extra > 2:
                base -= 0.25
            base -= 0.05  # 整体降权
        # 中间夹的"的、和、与"过多，说明断句不合适
        if gap_clean.count("的") >= 2:
            base -= 0.10
        return max(0.05, min(1.0, base))
