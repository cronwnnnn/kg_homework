"""知识图谱内存视图：把三元组 CSV 加载为 networkx 图 + 倒排索引。

提供：
    - load_triples(path)            原始三元组行（带元信息）
    - KGStore(triples)              图谱接口
        .graph                      networkx.MultiDiGraph
        .all_entities()             所有实体列表
        .neighbors(entity, hops)    K 跳邻居子图
        .triples_of(entity)         以该实体为头或尾的三元组
        .triples_with_relation(rel) 某种关系的三元组
        .find_path(a, b, k)         两实体间最多 k 跳的路径
        .related(a, b)              两实体间所有直接关系
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

import networkx as nx


@dataclass(frozen=True)
class TripleRow:
    head: str
    relation: str
    tail: str
    score: float = 1.0
    source: str = ""
    chapter: str = ""
    sentence: str = ""
    trigger: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


def load_triples(path: str) -> list[TripleRow]:
    """从 CSV 读取三元组（兼容 head/relation/tail 或 头实体/关系/尾实体 表头）。"""
    rows: list[TripleRow] = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            head = (r.get("head") or r.get("头实体") or "").strip()
            rel = (r.get("relation") or r.get("关系") or "").strip()
            tail = (r.get("tail") or r.get("尾实体") or "").strip()
            if not head or not rel or not tail or head == tail:
                continue
            try:
                sc = float((r.get("score") or "1").strip())
            except ValueError:
                sc = 1.0
            rows.append(
                TripleRow(
                    head=head,
                    relation=rel,
                    tail=tail,
                    score=sc,
                    source=(r.get("source") or "").strip(),
                    chapter=(r.get("chapter") or "").strip(),
                    sentence=(r.get("sentence") or "").strip(),
                    trigger=(r.get("trigger") or "").strip(),
                )
            )
    return rows


class KGStore:
    """以三元组列表初始化，构建 networkx 图与倒排索引。"""

    def __init__(self, triples: Iterable[TripleRow]) -> None:
        self.triples: list[TripleRow] = list(triples)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._by_head: dict[str, list[TripleRow]] = defaultdict(list)
        self._by_tail: dict[str, list[TripleRow]] = defaultdict(list)
        self._by_relation: dict[str, list[TripleRow]] = defaultdict(list)
        self._by_pair: dict[tuple[str, str], list[TripleRow]] = defaultdict(list)
        self._all_entities: set[str] = set()

        for t in self.triples:
            self.graph.add_edge(
                t.head, t.tail,
                label=t.relation,
                score=t.score,
                source=t.source,
                chapter=t.chapter,
                sentence=t.sentence,
                trigger=t.trigger,
            )
            self._by_head[t.head].append(t)
            self._by_tail[t.tail].append(t)
            self._by_relation[t.relation].append(t)
            self._by_pair[(t.head, t.tail)].append(t)
            self._all_entities.add(t.head)
            self._all_entities.add(t.tail)

    def all_entities(self) -> list[str]:
        return sorted(self._all_entities)

    def all_relations(self) -> list[str]:
        return sorted(self._by_relation)

    def triples_of(self, entity: str) -> list[TripleRow]:
        """实体作为头或尾出现的三元组集合（去重）。"""
        seen: set[tuple[str, str, str]] = set()
        out: list[TripleRow] = []
        for t in self._by_head.get(entity, []):
            if t.key not in seen:
                seen.add(t.key)
                out.append(t)
        for t in self._by_tail.get(entity, []):
            if t.key not in seen:
                seen.add(t.key)
                out.append(t)
        return out

    def out_edges(self, entity: str, relation: str | None = None) -> list[TripleRow]:
        rows = self._by_head.get(entity, [])
        if relation is None:
            return list(rows)
        return [t for t in rows if t.relation == relation]

    def in_edges(self, entity: str, relation: str | None = None) -> list[TripleRow]:
        rows = self._by_tail.get(entity, [])
        if relation is None:
            return list(rows)
        return [t for t in rows if t.relation == relation]

    def triples_with_relation(self, relation: str) -> list[TripleRow]:
        return list(self._by_relation.get(relation, []))

    def related(self, a: str, b: str) -> list[TripleRow]:
        """两实体间任意方向的直接关系。"""
        return list(self._by_pair.get((a, b), [])) + list(self._by_pair.get((b, a), []))

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        max_paths: int = 6,
    ) -> list[list[TripleRow]]:
        """简单 BFS 找路径：忽略方向，限制最长 max_hops。

        返回三元组序列列表，每条路径不重复经过同一节点。
        """
        if source not in self.graph or target not in self.graph:
            return []
        und = self.graph.to_undirected(as_view=True)
        out: list[list[TripleRow]] = []
        try:
            for path in nx.all_simple_paths(und, source, target, cutoff=max_hops):
                if len(out) >= max_paths:
                    break
                step: list[TripleRow] = []
                ok = True
                for u, v in zip(path, path[1:]):
                    edges = self._by_pair.get((u, v)) or self._by_pair.get((v, u))
                    if not edges:
                        ok = False
                        break
                    # 选 score 最高的边作为代表
                    step.append(max(edges, key=lambda t: t.score))
                if ok and step:
                    out.append(step)
        except nx.NetworkXNoPath:
            pass
        return out

    def neighbors_subgraph(
        self,
        entity: str,
        hops: int = 2,
        max_nodes: int = 60,
    ) -> nx.MultiDiGraph:
        """以实体为中心的 BFS 子图。"""
        if entity not in self.graph:
            return nx.MultiDiGraph()
        nodes: set[str] = {entity}
        frontier: set[str] = {entity}
        for _ in range(hops):
            if len(nodes) >= max_nodes:
                break
            nxt: set[str] = set()
            for u in frontier:
                for v in self.graph.successors(u):
                    if v not in nodes:
                        nxt.add(v)
                for v in self.graph.predecessors(u):
                    if v not in nodes:
                        nxt.add(v)
            frontier = nxt
            for x in list(frontier):
                nodes.add(x)
                if len(nodes) >= max_nodes:
                    break
        return self.graph.subgraph(nodes).copy()

    def iter_triples(self) -> Iterator[TripleRow]:
        return iter(self.triples)

    def __len__(self) -> int:
        return len(self.triples)
