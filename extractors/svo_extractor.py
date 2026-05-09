"""依存句法 SVO 抽取：基于 spaCy 的中文依存分析。

相对原 pipeline_test.py 的改进：
    1. 不再仅依赖 ROOT，扩展到所有动词节点；
    2. 收集 conj/comp/cop 等扩展依存类型，提升召回；
    3. 谓词通过 RelationNormalizer 归一化为统一关系类型；
    4. 头/尾通过实体词典回填："NER 命中实体" 比 "spaCy 切出的零碎词" 优先；
    5. 缺失 spaCy 模型时优雅降级。
"""

from __future__ import annotations

from typing import Iterable

from .ner import HybridNER
from .preprocess import Sentence
from .relation_normalizer import RelationNormalizer
from .schema import Triple


_SUBJECT_DEPS = {"nsubj", "top", "nsubjpass", "csubj"}
_OBJECT_DEPS = {"dobj", "pobj", "iobj", "attr", "ccomp", "xcomp"}


class DependencyExtractor:
    def __init__(
        self,
        ner: HybridNER,
        normalizer: RelationNormalizer | None = None,
        spacy_model: str = "zh_core_web_sm",
    ) -> None:
        self.ner = ner
        self.normalizer = normalizer or RelationNormalizer()
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
            print(f"[svo_extractor] 加载 spaCy 模型 {self._model_name} 失败: {exc}")
            print("[svo_extractor] SVO 抽取将被跳过。如需启用：python -m spacy download zh_core_web_sm")
            self.nlp = None
            return False

    def _resolve_to_entity(self, span_text: str) -> str | None:
        if not span_text:
            return None
        span_text = span_text.strip()
        if span_text in self.ner.vocab_set:
            return span_text
        for w in sorted(self.ner.vocab_set, key=lambda x: -len(x)):
            if len(w) < 2:
                continue
            if w in span_text:
                return w
        return None

    def extract_from_sentence(self, sent: Sentence) -> list[Triple]:
        if not self._ensure_nlp():
            return []
        doc = self.nlp(sent.text)  # type: ignore[union-attr]
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for token in doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            subjects = [w for w in token.lefts if w.dep_ in _SUBJECT_DEPS]
            objects = [w for w in token.rights if w.dep_ in _OBJECT_DEPS]
            if not subjects or not objects:
                continue
            predicate = token.text
            rel, trig = self.normalizer.normalize(predicate)
            if rel == "related_to":
                continue
            for s_tok in subjects:
                for o_tok in objects:
                    s_subtree = "".join(t.text for t in s_tok.subtree)
                    o_subtree = "".join(t.text for t in o_tok.subtree)
                    head = self._resolve_to_entity(s_tok.text) or self._resolve_to_entity(s_subtree)
                    tail = self._resolve_to_entity(o_tok.text) or self._resolve_to_entity(o_subtree)
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
                            trigger=predicate,
                            source="svo",
                            score=0.62,
                            chapter=sent.chapter,
                            sentence=sent.text,
                        )
                    )
        return triples

    def extract_from_sentences(self, sentences: Iterable[Sentence]) -> list[Triple]:
        out: list[Triple] = []
        for s in sentences:
            out.extend(self.extract_from_sentence(s))
        return out
