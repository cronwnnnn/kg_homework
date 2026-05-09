"""关系归一化：把自由的中文谓词映射到统一关系本体。"""

from __future__ import annotations

from .schema import RelationOntology


class RelationNormalizer:
    """关系类型归一化。

    使用方式：
        norm = RelationNormalizer()
        rel, trig = norm.normalize("提升")  # -> ("improves", "提升")
        rel, trig = norm.normalize("是一种")  # -> ("is_a", "是一种")
    """

    def __init__(self) -> None:
        self._exact_map: dict[str, str] = {}
        self._substring_map: list[tuple[str, str]] = []
        for trig, rel in RelationOntology.all_triggers():
            self._exact_map[trig] = rel
            self._substring_map.append((trig, rel))

    def normalize(self, predicate: str) -> tuple[str, str]:
        """返回 (relation_type, matched_trigger)。

        若都不命中，返回 ("related_to", predicate)。
        """
        p = (predicate or "").strip()
        if not p:
            return "related_to", ""
        if p in self._exact_map:
            return self._exact_map[p], p
        for trig, rel in self._substring_map:
            if trig in p:
                return rel, trig
        return "related_to", p

    def is_known_relation(self, predicate: str) -> bool:
        rel, _ = self.normalize(predicate)
        return rel != "related_to"
