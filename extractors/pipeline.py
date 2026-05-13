"""抽取流水线编排：串联预处理、NER、各类抽取器、归一化、LLM 增强。"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .llm_enhancer import LLMEnhancer
from .ner import HybridNER
from .numeric_extractor import NumericExtractor
from .pattern_extractor import PatternExtractor
from .preprocess import Paragraph, Sentence, TextPreprocessor
from .relation_normalizer import RelationNormalizer
from .schema import Triple
from .svo_extractor import DependencyExtractor
from .trigger_extractor import TriggerCooccurrenceExtractor
from .type_extractor import ChapterMembershipExtractor, TypeBasedExtractor, collect_cooccurring_entities


@dataclass
class PipelineConfig:
    use_trigger: bool = True
    use_pattern: bool = True
    use_svo: bool = False  # 默认关闭：spaCy 模型未装时无产出；装上后噪声较大，无正向贡献
    use_numeric: bool = True
    use_type: bool = True
    use_chapter: bool = True
    use_llm: bool = True

    llm_mode: str = "mock"  # "mock" | "openai"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    trigger_max_window: int = 30
    trigger_min_score: float = 0.55  # 调高（原 0.38）：FP 主要来自低分 trigger，详见审计
    spacy_model: str = "zh_core_web_sm"
    # 全局 score 阈值：所有 source 输出后统一过滤；为 0 时不过滤。
    # type/numeric/pattern/instance_of 等高质量来源 score 通常 ≥ 0.6，几乎不受影响。
    final_min_score: float = 0.55

    # 二阶段 LLM：在已有候选上根据正文补充「字面可核对」的新三元组（仅 openai 模式生效）
    use_llm_discovery: bool = False
    llm_discovery_max_chars: int = 16000
    llm_discovery_max_new_per_chapter: int = 40
    llm_discovery_max_existing_lines: int = 60

    # 论文内实体挖掘：默认关闭。原模块已归档到 extractors/archive/paper_entity_recognizer.py。
    # 实测发现长复合实体会"吞并"短实体造成 FP，主基线弃用。
    mine_paper_entities: bool = False
    paper_entity_min_doc_freq: int = 1  # ≥2 可抑制一次性噪声词

    output_dir: str = "output"
    csv_simple_path: str = "knowledge_graph.csv"
    csv_meta_path: str = "output/triples_with_meta.csv"
    entities_path: str = "output/entities.csv"
    stats_path: str = "output/extraction_stats.txt"


@dataclass
class PipelineResult:
    triples: list[Triple]
    entities: list[str]
    stats: dict[str, int | float | dict[str, int]]


class ExtractionPipeline:
    """端到端的中文知识图谱抽取流水线。"""

    def __init__(
        self,
        vocab: Iterable[str],
        config: PipelineConfig | None = None,
        entities_by_type: dict[str, list[str]] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = TextPreprocessor()
        self.ner = HybridNER(vocab=vocab, enable_numeric=True, enable_aircraft_code=True)
        self.normalizer = RelationNormalizer()
        self.trigger = TriggerCooccurrenceExtractor(
            self.ner,
            self.normalizer,
            max_window=self.config.trigger_max_window,
            min_score=self.config.trigger_min_score,
        )
        self.pattern = PatternExtractor(self.ner)
        self.numeric = NumericExtractor(self.ner)
        self.svo = DependencyExtractor(self.ner, self.normalizer, spacy_model=self.config.spacy_model)
        self.entities_by_type: dict[str, list[str]] = entities_by_type or {}
        self.type_ext = TypeBasedExtractor(self.entities_by_type) if self.entities_by_type else None
        self.chapter_ext = (
            ChapterMembershipExtractor(self.ner, self.entities_by_type, min_occur=2)
            if self.entities_by_type
            else None
        )
        self.llm = LLMEnhancer(
            mode=self.config.llm_mode,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
        )

    # ---------- 抽取 ----------

    def run(self, raw_text: str, verbose: bool = True) -> PipelineResult:
        paragraphs, sentences = self.preprocessor.process(raw_text)
        if verbose:
            print(f"[pipeline] 段落数={len(paragraphs)}  句子数={len(sentences)}")

        mined_list: list[str] = []
        paper_mined_added = 0
        if self.config.mine_paper_entities:
            # 该模块已归档到 extractors/archive/，主基线默认关闭。
            # 若要复用，从 archive 临时引入：
            import sys
            import os
            _archive = os.path.join(os.path.dirname(__file__), "archive")
            if _archive not in sys.path:
                sys.path.insert(0, _archive)
            from paper_entity_recognizer import PaperEntityRecognizer  # type: ignore

            before_vocab = len(self.ner.vocab_set)
            rec = PaperEntityRecognizer(spacy_model=self.config.spacy_model)
            mined_list = rec.collect_from_sentences(
                sentences,
                vocab_boost=self.ner.vocab_set,
                min_doc_freq=self.config.paper_entity_min_doc_freq,
            )
            self.ner.add_terms(mined_list)
            paper_mined_added = len(self.ner.vocab_set) - before_vocab
            if verbose:
                print(
                    f"[pipeline] 论文实体挖掘(spaCy): 候选串 {len(mined_list)}，"
                    f"新并入词典 {paper_mined_added} 个（词典总词 {len(self.ner.vocab_set)}）"
                )

        triples: list[Triple] = []
        per_source: Counter[str] = Counter()

        if self.config.use_trigger:
            tri = self.trigger.extract_from_sentences(sentences)
            triples.extend(tri)
            per_source["trigger"] = len(tri)
            if verbose:
                print(f"[pipeline] trigger 抽取: {len(tri)} 条")

        if self.config.use_pattern:
            pat = self.pattern.extract_from_sentences(sentences)
            triples.extend(pat)
            per_source["pattern"] = len(pat)
            if verbose:
                print(f"[pipeline] pattern 抽取: {len(pat)} 条")

        if self.config.use_numeric:
            num = self.numeric.extract_from_sentences(sentences)
            triples.extend(num)
            per_source["numeric"] = len(num)
            if verbose:
                print(f"[pipeline] numeric 抽取: {len(num)} 条")

        if self.config.use_svo:
            svo = self.svo.extract_from_sentences(sentences)
            triples.extend(svo)
            per_source["svo"] = len(svo)
            if verbose:
                print(f"[pipeline] svo 抽取: {len(svo)} 条")

        if self.config.use_chapter and self.chapter_ext is not None:
            ch = self.chapter_ext.extract_from_sentences(sentences)
            triples.extend(ch)
            per_source["chapter"] = len(ch)
            if verbose:
                print(f"[pipeline] chapter 抽取: {len(ch)} 条")

        # 收集"已观察到的实体集合"——只允许真正在文中出现的实体进入 type_ext。
        # 来源 1：其他抽取器已经产出的 triple 端点；
        # 来源 2：collect_cooccurring_entities —— 句子级类型对共现，弥补
        #         "只在共现中出现、未被关系抽取器命中"的实体（不产 triple，仅作标签来源）。
        observed_entities: set[str] = set()
        for t in triples:
            observed_entities.add(t.head)
            observed_entities.add(t.tail)
        if self.config.use_type and self.type_ext is not None:
            observed_entities |= collect_cooccurring_entities(
                self.ner, sentences, self.entities_by_type
            )
            self.type_ext.set_observed_entities(observed_entities)
            ty = self.type_ext.extract()
            triples.extend(ty)
            per_source["type"] = len(ty)
            if verbose:
                print(f"[pipeline] type 抽取: {len(ty)} 条")

        if verbose:
            print(f"[pipeline] 抽取小计（去重前）：{len(triples)} 条")
        triples = self._merge(triples)
        if verbose:
            print(f"[pipeline] 抽取小计（去重后）：{len(triples)} 条")

        # 全局 score 阈值过滤：剔除低质量来源（主要是 cooccur 0.5 与低分 trigger）。
        # 不过滤 numeric/pattern/instance_of/chapter 等高质量产出（它们 score 通常 ≥ 0.6）。
        if self.config.final_min_score > 0:
            n_before_filter = len(triples)
            triples = [t for t in triples if t.score >= self.config.final_min_score]
            if verbose:
                print(
                    f"[pipeline] 全局 score 过滤 (>= {self.config.final_min_score}): "
                    f"{n_before_filter} → {len(triples)} 条 "
                    f"（删 {n_before_filter - len(triples)} 条低质量）"
                )

        if self.config.use_llm:
            triples = self._llm_polish(triples, sentences, verbose=verbose)
            if verbose:
                print(f"[pipeline] LLM 增强后：{len(triples)} 条")

        if self.config.use_llm_discovery:
            discovered = self._llm_discover_novel(paragraphs, triples, verbose=verbose)
            if discovered:
                n_before = len(triples)
                triples.extend(discovered)
                triples = self._merge(triples)
                if verbose:
                    print(
                        f"[pipeline] LLM 发现补全：提交 {len(discovered)} 条候选，"
                        f"合并后总条数 {len(triples)}（合并前 {n_before}）"
                    )

        triples.sort(key=lambda t: (-t.score, t.relation, t.head, t.tail))

        entities = self._collect_entities(triples)

        stats: dict[str, int | float | dict[str, int]] = {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "triple_count": len(triples),
            "entity_count": len(entities),
            "by_source": dict(Counter(t.source for t in triples)),
            "by_relation": dict(Counter(t.relation for t in triples)),
            "extracted_per_source_before_merge": dict(per_source),
            "paper_mined_candidates": len(mined_list),
            "paper_mined_added_to_vocab": paper_mined_added,
        }
        return PipelineResult(triples=triples, entities=entities, stats=stats)

    # ---------- 合并 / 增强 ----------

    _ALLOWED_NUM_REL_TAIL = {
        "has_value", "greater_than_value", "less_than_value",
        "greater_than", "less_than", "equals_to",
    }

    @classmethod
    def _is_pure_number(cls, s: str) -> bool:
        if not s:
            return True
        return s.replace(".", "").replace("%", "").replace("-", "").isdigit()

    @classmethod
    def _merge(cls, triples: list[Triple]) -> list[Triple]:
        bucket: dict[tuple[str, str, str], Triple] = {}
        for t in triples:
            if not t.head or not t.tail or t.head == t.tail:
                continue
            head_is_num = cls._is_pure_number(t.head)
            tail_is_num = cls._is_pure_number(t.tail)
            if head_is_num:
                continue
            if tail_is_num and t.relation not in cls._ALLOWED_NUM_REL_TAIL:
                continue
            key = (t.head, t.relation, t.tail)
            if key not in bucket:
                bucket[key] = t
            else:
                cur = bucket[key]
                if t.score > cur.score:
                    bucket[key] = t
                elif t.source != cur.source:
                    bucket[key] = Triple(
                        head=cur.head,
                        relation=cur.relation,
                        tail=cur.tail,
                        trigger=cur.trigger or t.trigger,
                        source=f"{cur.source}+{t.source}",
                        score=max(cur.score, t.score),
                        chapter=cur.chapter or t.chapter,
                        sentence=cur.sentence or t.sentence,
                    )
        return list(bucket.values())

    def _llm_polish(
        self,
        triples: list[Triple],
        sentences: Sequence[Sentence],
        verbose: bool = True,
    ) -> list[Triple]:
        # 按章节分组打包送给 LLM；mock 模式下相当于"再去一遍重 + 关系归一化"
        by_chapter: dict[str, list[Triple]] = defaultdict(list)
        for t in triples:
            by_chapter[t.chapter].append(t)

        # 在 mock 模式下没必要按章节拆，直接整体处理一次即可
        if self.llm.mode == "mock":
            return self.llm.enhance(triples)

        out: list[Triple] = []
        polish_chars = int(os.environ.get("KG_LLM_POLISH_CHARS", "16000"))
        for chapter, group in by_chapter.items():
            chapter_text = "\n".join(s.text for s in sentences if s.chapter == chapter)[:polish_chars]
            ents = sorted({t.head for t in group} | {t.tail for t in group})
            polished = self.llm.enhance(group, paragraph_text=chapter_text, entities=ents)
            out.extend(polished)
            if verbose:
                print(f"[pipeline] LLM polish chapter={chapter or '(无章节)'}  in={len(group)}  out={len(polished)}")
        return self._merge(out)

    def _llm_discover_novel(
        self,
        paragraphs: Sequence[Paragraph],
        triples: Sequence[Triple],
        verbose: bool = True,
    ) -> list[Triple]:
        """按章节聚合正文，请求 LLM 补充未出现在候选中的、可在原文逐字核对的新三元组。"""
        if self.llm.mode != "openai":
            if verbose:
                print("[pipeline] LLM 发现补全已开启，但当前非 openai 模式，已跳过。")
            return []
        chapter_texts = self._aggregate_chapter_paragraphs(paragraphs)
        out: list[Triple] = []
        for chapter_key, text in sorted(chapter_texts.items(), key=lambda x: x[0] or "\uffff"):
            text = (text or "").strip()
            if len(text) < 40:
                continue
            disc_lim = int(os.environ.get("KG_LLM_DISCOVER_CHARS", str(self.config.llm_discovery_max_chars)))
            text = text[:disc_lim]
            existing = [t for t in triples if (t.chapter or "") == chapter_key]
            if verbose:
                print(
                    f"[pipeline] LLM 发现 chapter={chapter_key or '(无章节)'}  "
                    f"len={len(text)}  已有候选={len(existing)}"
                )
            out.extend(
                self.llm.discover_novel_triples(
                    chapter_text=text,
                    chapter=chapter_key,
                    existing_triples=existing,
                    max_new=self.config.llm_discovery_max_new_per_chapter,
                    max_existing_lines=self.config.llm_discovery_max_existing_lines,
                )
            )
        return out

    @staticmethod
    def _aggregate_chapter_paragraphs(paragraphs: Sequence[Paragraph]) -> dict[str, str]:
        """按章节键拼接该章所有段落正文。"""
        parts: dict[str, list[str]] = defaultdict(list)
        for p in paragraphs:
            key = p.chapter or ""
            parts[key].append(p.text)
        return {k: "\n".join(v) for k, v in parts.items()}

    # ---------- 辅助 ----------

    @staticmethod
    def _collect_entities(triples: Iterable[Triple]) -> list[str]:
        bag: set[str] = set()
        for t in triples:
            bag.add(t.head)
            bag.add(t.tail)
        return sorted(bag)

    # ---------- 输出 ----------

    def export(self, result: PipelineResult) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        # 1) 简版（兼容老格式 头实体,关系,尾实体）
        with open(self.config.csv_simple_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["头实体", "关系", "尾实体"])
            for t in result.triples:
                writer.writerow([t.head, t.relation, t.tail])

        # 2) 元信息版
        os.makedirs(os.path.dirname(self.config.csv_meta_path) or ".", exist_ok=True)
        with open(self.config.csv_meta_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["head", "relation", "tail", "trigger", "source", "score", "chapter", "sentence"])
            for t in result.triples:
                writer.writerow([t.head, t.relation, t.tail, t.trigger, t.source, f"{t.score:.3f}", t.chapter, t.sentence])

        # 3) 实体清单
        with open(self.config.entities_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entity"])
            for e in result.entities:
                writer.writerow([e])

        # 4) 统计报告
        with open(self.config.stats_path, "w", encoding="utf-8") as f:
            f.write("==== 抽取统计 ====\n")
            f.write(f"段落数: {result.stats['paragraph_count']}\n")
            f.write(f"句子数: {result.stats['sentence_count']}\n")
            f.write(f"实体数: {result.stats['entity_count']}\n")
            f.write(f"三元组数: {result.stats['triple_count']}\n\n")
            if result.stats.get("paper_mined_added_to_vocab", 0) or result.stats.get("paper_mined_candidates"):
                f.write("---- 论文实体挖掘 (spaCy) ----\n")
                f.write(f"  候选串数: {result.stats.get('paper_mined_candidates', 0)}\n")
                f.write(f"  新并入词典: {result.stats.get('paper_mined_added_to_vocab', 0)}\n\n")
            f.write("---- 按来源 (合并后) ----\n")
            for k, v in sorted(result.stats["by_source"].items(), key=lambda x: -x[1]):
                f.write(f"  {k}: {v}\n")
            f.write("\n---- 按关系类型 (Top-30) ----\n")
            top_relations = sorted(result.stats["by_relation"].items(), key=lambda x: -x[1])[:30]
            for k, v in top_relations:
                f.write(f"  {k}: {v}\n")
            f.write("\n---- 按来源 (合并前) ----\n")
            for k, v in sorted(result.stats["extracted_per_source_before_merge"].items(), key=lambda x: -x[1]):
                f.write(f"  {k}: {v}\n")
        print(f"[pipeline] 输出: {self.config.csv_simple_path}, {self.config.csv_meta_path}, {self.config.entities_path}, {self.config.stats_path}")
