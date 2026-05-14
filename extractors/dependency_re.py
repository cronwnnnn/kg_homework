"""依存句法增强的关系抽取器（DependencyREExtractor）。

相对原 svo_extractor.py 的核心改进：
    1. **不强求 head/tail 在词典内**：直接用子树拼接得到完整 NP（如"上下机翼相互干扰"）；
    2. **子树清洗**：去掉副词、助词、连词、标点等修饰词，得到干净的实体名；
    3. **触发词驱动归一化**：用 RelationOntology.TRIGGER_TABLE 把谓词映射到关系本体；
    4. **cop 模式**：识别"X 为 N 单位"的 has_value 子句；
    5. **否定/被动检测**：自动翻转 head/tail 或降分；
    6. **关系-类型约束**：复用 schema.py 的 RELATION_TYPE_CONSTRAINTS；
    7. **降级**：spaCy 模型缺失时优雅返回空列表，不破坏流水线。

依赖：
    spaCy + zh_core_web_sm（中文依存模型）。
    安装：python -m spacy download zh_core_web_sm
"""

from __future__ import annotations

import re
from typing import Iterable

from .ner import HybridNER
from .preprocess import Sentence
from .relation_normalizer import RelationNormalizer
from .schema import RelationOntology, Triple


_SUBJECT_DEPS = {"nsubj", "top", "nsubjpass", "csubj"}
_OBJECT_DEPS = {"dobj", "pobj", "iobj", "attr", "ccomp", "xcomp"}

_NEGATION_TOKENS = {"不", "未", "无", "没", "非", "勿", "莫", "毋"}
_PASSIVE_TOKENS = {"被", "由", "受"}

_HEAD_ROLE_FILTER = {"AUX", "PART", "PUNCT", "CCONJ", "SCONJ"}
_DROP_TOKENS = {"的", "了", "着", "过", "地", "得", "之", "且", "并", "及", "或", "和", "与"}
_DROP_MODAL = {"会", "能", "能够", "可", "可以", "应", "应当", "应该", "需要", "必须"}

_PRONOUN_HEAD = {
    "这", "这些", "那", "那些", "它", "它们", "他", "他们", "其", "本", "本节",
    "本章", "上述", "下述", "前述", "如下", "以下", "如上", "此外", "另外",
    "其中", "我们", "我", "你", "你们",
}

_PREP_PREFIX_PATTERNS = [
    re.compile(r"^(在|对|经|通过|借助|利用|采用|使用)[^一-龥A-Za-z]{0,5}"),
    re.compile(r"^(为|为了|因为|由于|根据|按照|依据|针对)[^一-龥A-Za-z]{0,5}"),
]

_GENERIC_HEAD = {
    "研究内容", "本研究", "本工作", "本章", "本节", "本文", "结果", "结论",
    "讨论", "分析", "目标", "目的", "方法", "本方法",
    "过程", "中", "时", "前", "后", "上", "下", "内", "外",
    "情况", "状态", "方式", "情形", "条件",
    "以上分析结果", "上述分析", "分析结果", "结果显示",
    "变体过程中", "研究过程中", "试验过程中", "仿真过程中",
}

_FUZZY_HEAD_PATTERNS = [
    re.compile(r"过程中$"),
    re.compile(r"^于"),
    re.compile(r"^以上"),
    re.compile(r"^上述"),
    re.compile(r"^下述"),
    re.compile(r"^如下"),
]

_NUMERIC_TAIL_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*"
    r"(?:km/h|m/s|kt|kg|t|g|kw|kW|mw|MW|cm|mm|m|km|s|ms|min|h|N·m|N\u00b7m|Nm|N|"
    r"度|°|°C|%|Mach|马赫|克|吨|公斤|公里|米|秒|分钟|小时|赫兹|Hz|Pa|kPa|MPa)?\s*$"
)

_MIN_NP_LEN = 2
_MAX_NP_LEN = 25


class DependencyREExtractor:
    """基于 spaCy 依存树的关系抽取器。"""

    def __init__(
        self,
        ner: HybridNER,
        normalizer: RelationNormalizer | None = None,
        spacy_model: str = "zh_core_web_sm",
        min_score: float = 0.55,
        prefer_dict_entity: bool = True,
        require_dict_head: bool = False,
        require_dict_tail: bool = False,
    ) -> None:
        self.ner = ner
        self.normalizer = normalizer or RelationNormalizer()
        self.min_score = min_score
        self.prefer_dict_entity = prefer_dict_entity
        self.require_dict_head = require_dict_head
        self.require_dict_tail = require_dict_tail
        self.nlp = None
        self._model_name = spacy_model
        self._tried_load = False

    def _ensure_nlp(self) -> bool:
        if self._tried_load:
            return self.nlp is not None
        self._tried_load = True
        try:
            import spacy  # type: ignore

            self.nlp = spacy.load(self._model_name)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[dependency_re] 加载 spaCy 模型 {self._model_name} 失败: {exc}")
            print(
                "[dependency_re] 依存抽取将被跳过。如需启用："
                "python -m spacy download zh_core_web_sm"
            )
            self.nlp = None
            return False

    def extract_from_sentences(self, sentences: Iterable[Sentence]) -> list[Triple]:
        if not self._ensure_nlp():
            return []
        out: list[Triple] = []
        for s in sentences:
            out.extend(self.extract_from_sentence(s))
        return out

    def extract_from_sentence(self, sent: Sentence) -> list[Triple]:
        if not self._ensure_nlp():
            return []
        doc = self.nlp(sent.text)  # type: ignore[union-attr]
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()

        for token in doc:
            if token.pos_ == "VERB":
                triples.extend(self._extract_from_verb(token, sent, seen))
            elif token.dep_ == "cop":
                triples.extend(self._extract_from_cop(token, sent, seen))

        return triples

    def _extract_from_verb(self, verb, sent: Sentence, seen: set) -> list[Triple]:
        rel, trig = self.normalizer.normalize(verb.text)
        if rel == "related_to":
            return []

        subjects = [w for w in verb.children if w.dep_ in _SUBJECT_DEPS]
        objects = [w for w in verb.children if w.dep_ in _OBJECT_DEPS]

        if not objects:
            return []
        if not subjects:
            for ancestor in verb.ancestors:
                cand_subjects = [
                    w for w in ancestor.children if w.dep_ in _SUBJECT_DEPS
                ]
                if cand_subjects:
                    subjects = cand_subjects
                    break
        if not subjects:
            return []

        negated = self._has_negation(verb)
        passive = self._has_passive(verb)

        out: list[Triple] = []
        for s_tok in subjects:
            head_text = self._clean_subtree(s_tok)
            if not self._valid_np(head_text):
                continue
            for o_tok in objects:
                tail_text = self._clean_subtree(o_tok, allow_numeric=True)
                if not self._valid_np(tail_text, allow_numeric=True):
                    continue

                head_canon = self._canonicalize(head_text)
                tail_canon = self._canonicalize(tail_text, allow_numeric=True)
                if not head_canon or not tail_canon or head_canon == tail_canon:
                    continue

                if self.require_dict_head and head_canon not in self.ner.vocab_set:
                    continue
                if self.require_dict_tail and tail_canon not in self.ner.vocab_set:
                    if not _NUMERIC_TAIL_RE.match(tail_canon):
                        continue

                if self._tail_violates_constraint(rel, tail_canon):
                    continue

                final_head, final_tail = (
                    (tail_canon, head_canon) if passive else (head_canon, tail_canon)
                )

                key = (final_head, rel, final_tail)
                if key in seen:
                    continue
                seen.add(key)

                score = self._score(
                    head_text=head_canon,
                    tail_text=tail_canon,
                    rel=rel,
                    negated=negated,
                    passive=passive,
                    subtree_match=(head_canon == head_text or tail_canon == tail_text),
                )
                if score < self.min_score:
                    continue

                out.append(
                    Triple(
                        head=final_head,
                        relation=rel,
                        tail=final_tail,
                        trigger=trig or verb.text,
                        source="dep",
                        score=score,
                        chapter=sent.chapter,
                        sentence=sent.text,
                    )
                )
        return out

    def _extract_from_cop(self, cop_tok, sent: Sentence, seen: set) -> list[Triple]:
        head_node = cop_tok.head
        subjects = [w for w in head_node.children if w.dep_ in _SUBJECT_DEPS]
        if not subjects:
            return []

        head_text = self._clean_subtree(head_node, allow_numeric=True)
        if not head_text:
            return []

        is_numeric_tail = bool(_NUMERIC_TAIL_RE.match(head_text)) or any(
            c.isdigit() for c in head_text
        )
        if is_numeric_tail:
            rel = "has_value"
        else:
            rel = "is_a"

        out: list[Triple] = []
        for s_tok in subjects:
            head_np = self._clean_subtree(s_tok)
            tail_np = head_text
            head_canon = self._canonicalize(head_np)
            tail_canon = self._canonicalize(tail_np, allow_numeric=is_numeric_tail)
            if not head_canon or not tail_canon or head_canon == tail_canon:
                continue
            if self._tail_violates_constraint(rel, tail_canon):
                continue

            key = (head_canon, rel, tail_canon)
            if key in seen:
                continue
            seen.add(key)

            score = 0.72 if is_numeric_tail else 0.66
            if score < self.min_score:
                continue
            out.append(
                Triple(
                    head=head_canon,
                    relation=rel,
                    tail=tail_canon,
                    trigger=cop_tok.text,
                    source="dep",
                    score=score,
                    chapter=sent.chapter,
                    sentence=sent.text,
                )
            )
        return out

    @staticmethod
    def _has_negation(verb) -> bool:
        for child in verb.children:
            if child.text in _NEGATION_TOKENS:
                return True
        for left in verb.lefts:
            if left.text in _NEGATION_TOKENS:
                return True
            for sub in left.lefts:
                if sub.text in _NEGATION_TOKENS:
                    return True
        return False

    @staticmethod
    def _has_passive(verb) -> bool:
        for child in verb.children:
            if child.text in _PASSIVE_TOKENS:
                return True
            if child.dep_ == "nsubjpass":
                return True
        return False

    @staticmethod
    def _clean_subtree(token, allow_numeric: bool = False) -> str:
        if token is None:
            return ""
        tokens_in_order = sorted(token.subtree, key=lambda t: t.i)

        kept: list[str] = []
        for t in tokens_in_order:
            if t.pos_ == "PUNCT":
                continue
            if t.pos_ == "AUX" and t.text in _DROP_MODAL:
                continue
            if t.text in _DROP_TOKENS:
                continue
            if t.text in _DROP_MODAL:
                continue
            if not allow_numeric and t.pos_ == "NUM":
                if not (len(t.text) >= 2 and any("\u4e00" <= c <= "\u9fa5" for c in t.text)):
                    continue
            if t.pos_ in {"ADV"} and len(t.text) <= 2 and not kept:
                continue
            kept.append(t.text)

        out = "".join(kept).strip()
        while out and out[0] in "的了着过地得之且并及或和与，。；：、 ":
            out = out[1:]
        while out and out[-1] in "的了着过地得之且并及或和与，。；：、 ":
            out = out[:-1]
        return out

    def _canonicalize(self, text: str, allow_numeric: bool = False) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        for pat in _PREP_PREFIX_PATTERNS:
            text = pat.sub("", text)
        text = text.strip()
        if not text:
            return ""

        if text in _PRONOUN_HEAD:
            return ""
        if text in _GENERIC_HEAD:
            return ""
        for pat in _FUZZY_HEAD_PATTERNS:
            if pat.search(text):
                return ""

        if self.prefer_dict_entity:
            if text in self.ner.vocab_set:
                return text
            best: str | None = None
            for w in self.ner.vocab_set:
                if len(w) < 2:
                    continue
                if w in text:
                    if best is None or len(w) > len(best):
                        best = w
            if best and len(best) >= max(2, len(text) // 2):
                return best
            if allow_numeric and _NUMERIC_TAIL_RE.match(text):
                return text
            return ""
        if allow_numeric and _NUMERIC_TAIL_RE.match(text):
            return text
        return text

    @staticmethod
    def _valid_np(text: str, allow_numeric: bool = False) -> bool:
        if not text:
            return False
        if len(text) < _MIN_NP_LEN or len(text) > _MAX_NP_LEN:
            return False
        if not allow_numeric and text.replace(".", "").isdigit():
            return False
        if text in _PRONOUN_HEAD or text in _GENERIC_HEAD:
            return False
        return True

    @staticmethod
    def _tail_violates_constraint(rel: str, tail: str) -> bool:
        constraints = RelationOntology.RELATION_TYPE_CONSTRAINTS.get(rel)
        if not constraints:
            return False
        forbid = constraints.get("tail_forbid", set())
        if "numeric" in forbid and _NUMERIC_TAIL_RE.match(tail):
            return True
        return False

    def _score(
        self,
        head_text: str,
        tail_text: str,
        rel: str,
        negated: bool,
        passive: bool,
        subtree_match: bool,
    ) -> float:
        score = 0.72

        if head_text in self.ner.vocab_set:
            score += 0.04
        if tail_text in self.ner.vocab_set:
            score += 0.04

        if 2 <= len(head_text) <= 8 and 2 <= len(tail_text) <= 8:
            score += 0.03

        if len(head_text) > 14 or len(tail_text) > 14:
            score -= 0.10

        if negated:
            score -= 0.20
        if passive and rel in {"manufactures", "develops", "uses_method"}:
            score += 0.02

        return max(0.0, min(1.0, score))
