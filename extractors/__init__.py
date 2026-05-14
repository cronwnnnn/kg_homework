"""
中文知识图谱多策略抽取流水线。

设计原则：
1. 传统 NLP（词典 NER + 触发词共现 + 关系模板 + 依存 SVO + 数值正则）做主力召回；
2. LLM 仅做"过滤 / 关系归一化 / 隐式补全"，不允许独立产出三元组；
3. 每条三元组都带 `source` 与 `score`，便于后续评估与回溯。
"""

from .schema import Triple, Mention, RelationOntology
from .preprocess import TextPreprocessor
from .ner import HybridNER
from .trigger_extractor import TriggerCooccurrenceExtractor
from .pattern_extractor import PatternExtractor
from .svo_extractor import DependencyExtractor
from .dependency_re import DependencyREExtractor
from .numeric_extractor import NumericExtractor
from .relation_normalizer import RelationNormalizer
from .llm_enhancer import LLMEnhancer
from .type_extractor import TypeBasedExtractor, ChapterMembershipExtractor
from .pipeline import ExtractionPipeline

__all__ = [
    "Triple",
    "Mention",
    "RelationOntology",
    "TextPreprocessor",
    "HybridNER",
    "TriggerCooccurrenceExtractor",
    "PatternExtractor",
    "DependencyExtractor",
    "DependencyREExtractor",
    "NumericExtractor",
    "RelationNormalizer",
    "LLMEnhancer",
    "TypeBasedExtractor",
    "ChapterMembershipExtractor",
    "ExtractionPipeline",
]
