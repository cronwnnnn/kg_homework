"""知识图谱简易浏览应用（Streamlit + pyvis）。

依赖（可选）：
    uv sync --extra app

运行：
    uv run streamlit run app_streamlit.py

默认加载 output/triples_with_meta.csv；若不存在会提示先运行 run_extract.py。
"""

from __future__ import annotations

import csv
import os

import networkx as nx
import streamlit as st
from pyvis.network import Network


def load_triples(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def build_subgraph(rows: list[dict[str, str]], center: str, max_nodes: int, min_score: float) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for row in rows:
        try:
            sc = float(row.get("score", "1") or 1)
        except ValueError:
            sc = 1.0
        if sc < min_score:
            continue
        h = row.get("head") or row.get("头实体") or ""
        r = row.get("relation") or row.get("关系") or ""
        t = row.get("tail") or row.get("尾实体") or ""
        if not h or not r or not t:
            continue
        g.add_edge(h, t, label=r, score=sc)

    if center not in g:
        return nx.MultiDiGraph()

    nodes = {center}
    frontier = {center}
    while frontier and len(nodes) < max_nodes:
        nxt: set[str] = set()
        for u in frontier:
            for v in g.successors(u):
                if v not in nodes:
                    nxt.add(v)
            for v in g.predecessors(u):
                if v not in nodes:
                    nxt.add(v)
        frontier = {x for x in nxt if x not in nodes}
        for x in frontier:
            nodes.add(x)
            if len(nodes) >= max_nodes:
                break
    return g.subgraph(nodes).copy()


def to_pyvis_html(g: nx.MultiDiGraph) -> str:
    net = Network(height="520px", width="100%", directed=True, bgcolor="#111", font_color="#eee")
    net.barnes_hut()
    for n in g.nodes():
        net.add_node(n, label=n, title=n, color="#4a9eff")
    for u, v, key, data in g.edges(keys=True, data=True):
        lab = data.get("label", "")
        net.add_edge(u, v, title=lab, label=lab[:16] + ("…" if len(lab) > 16 else ""))
    return net.generate_html()


def main() -> None:
    st.set_page_config(page_title="变体飞行器知识图谱", layout="wide")
    st.title("单-双折叠翼变体飞行器 · 知识图谱浏览")

    default_csv = os.path.join("output", "triples_with_meta.csv")
    path = st.sidebar.text_input("三元组 CSV 路径", value=default_csv)
    min_score = st.sidebar.slider("最小 score", 0.0, 1.0, 0.35, 0.05)
    max_nodes = st.sidebar.slider("子图最大节点数", 10, 200, 60, 5)

    if not os.path.isfile(path):
        st.error(f"找不到文件：{path}。请先运行：`uv run python run_extract.py`")
        return

    rows = load_triples(path)
    st.sidebar.caption(f"已加载 {len(rows)} 行（含表头外数据）")

    entities: set[str] = set()
    for row in rows:
        h = row.get("head") or row.get("头实体") or ""
        t = row.get("tail") or row.get("尾实体") or ""
        if h:
            entities.add(h)
        if t:
            entities.add(t)

    if not entities:
        st.warning("未解析到任何实体列，请确认 CSV 含 head/tail 或 头实体/尾实体。")
        return
    center = st.sidebar.selectbox("中心实体", options=sorted(entities))

    g = build_subgraph(rows, center, max_nodes=max_nodes, min_score=min_score)
    if g.number_of_nodes() == 0:
        st.warning("该中心在过滤后的图中无邻接边，请换实体或降低 score。")
        return

    html = to_pyvis_html(g)
    st.components.v1.html(html, height=560, scrolling=True)
    st.caption(f"子图节点数 {g.number_of_nodes()}，边数 {g.number_of_edges()}")


if __name__ == "__main__":
    main()
