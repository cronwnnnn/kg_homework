
from __future__ import annotations

import csv
import json
import os
from collections import Counter

import networkx as nx
import streamlit as st
from pyvis.network import Network


_DEFAULT_CSV = os.path.join("output", "triples_with_meta.csv")
_ENTITY_TYPE_JSON = os.path.join("data", "entities_by_type.json")

# 节点按类型染色
_TYPE_COLOR: dict[str, str] = {
    "AIRCRAFT": "#4a9eff",
    "WING_CONFIGURATION": "#3ad29f",
    "PARAMETER": "#f8c537",
    "AERODYNAMIC_CONCEPT": "#ff7f50",
    "STRUCTURAL_COMPONENT": "#9b59b6",
    "CONTROL_METHOD": "#e74c3c",
    "PERFORMANCE_METRIC": "#1abc9c",
    "ORGANIZATION": "#34495e",
    "PERSON": "#e67e22",
    "TECHNOLOGY": "#2980b9",
    "MATERIAL": "#16a085",
    "FLIGHT_PHASE": "#d35400",
    "EQUATION": "#7f8c8d",
    "CONCEPT": "#c0392b",
    "_UNKNOWN_": "#bbbbbb",
}


@st.cache_data(show_spinner=False)
def load_triples(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


@st.cache_data(show_spinner=False)
def load_entity_types(path: str) -> dict[str, str]:
    """返回 实体名 → 类型名 的索引。"""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for type_name, items in data.items():
            if not isinstance(items, list):
                continue
            for w in items:
                w = (w or "").strip()
                if w:
                    out.setdefault(w, type_name)
    return out


def _row_get(row: dict[str, str], *keys: str) -> str:
    for k in keys:
        v = row.get(k, "").strip() if row.get(k) else ""
        if v:
            return v
    return ""


def build_subgraph(
    rows: list[dict[str, str]],
    center: str,
    max_nodes: int,
    min_score: float,
    rel_whitelist: set[str] | None,
    min_text_len: int,
    max_hops: int,
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for row in rows:
        try:
            sc = float(row.get("score", "1") or 1)
        except ValueError:
            sc = 1.0
        if sc < min_score:
            continue
        h = _row_get(row, "head", "头实体")
        r = _row_get(row, "relation", "关系")
        t = _row_get(row, "tail", "尾实体")
        if not h or not r or not t:
            continue
        if rel_whitelist is not None and r not in rel_whitelist:
            continue
        if len(h) < min_text_len or len(t) < min_text_len:
            continue
        g.add_edge(h, t, label=r, score=sc, source=row.get("source", ""))

    if center not in g:
        return nx.MultiDiGraph()

    nodes = {center}
    frontier = {center}
    for _ in range(max_hops):
        if len(nodes) >= max_nodes:
            break
        nxt: set[str] = set()
        for u in frontier:
            for v in list(g.successors(u)) + list(g.predecessors(u)):
                if v not in nodes:
                    nxt.add(v)
        frontier = {x for x in nxt if x not in nodes}
        for x in frontier:
            nodes.add(x)
            if len(nodes) >= max_nodes:
                break
    return g.subgraph(nodes).copy()


def to_pyvis_html(g: nx.MultiDiGraph, ent_types: dict[str, str], center: str) -> str:
    net = Network(
        height="600px", width="100%", directed=True,
        bgcolor="#121212", font_color="#eeeeee",
    )
    net.barnes_hut(gravity=-7000, central_gravity=0.3, spring_length=120, spring_strength=0.05)
    for n in g.nodes():
        type_name = ent_types.get(n) or "_UNKNOWN_"
        color = _TYPE_COLOR.get(type_name, _TYPE_COLOR["_UNKNOWN_"])
        is_center = (n == center)
        net.add_node(
            n,
            label=n,
            title=f"{n}\n类型: {type_name if type_name != '_UNKNOWN_' else '未识别'}",
            color={"background": color, "border": "#ffffff" if is_center else "#333"},
            size=28 if is_center else 18,
            font={"size": 16 if is_center else 12, "color": "#eee"},
        )
    for u, v, _key, data in g.edges(keys=True, data=True):
        lab = data.get("label", "")
        sc = data.get("score", 1.0)
        src = data.get("source", "")
        title = f"{u} --[{lab}]--> {v}\nscore={sc:.2f}  source={src}"
        net.add_edge(
            u, v,
            title=title,
            label=lab[:16] + ("…" if len(lab) > 16 else ""),
            font={"size": 10, "color": "#cccccc"},
            arrows={"to": {"enabled": True}},
        )
    return net.generate_html(notebook=False)


def _stats_panel(rows: list[dict[str, str]]) -> None:
    rel_cnt: Counter[str] = Counter()
    src_cnt: Counter[str] = Counter()
    ents: set[str] = set()
    for r in rows:
        h = _row_get(r, "head", "头实体")
        t = _row_get(r, "tail", "尾实体")
        if h:
            ents.add(h)
        if t:
            ents.add(t)
        rel = _row_get(r, "relation", "关系")
        if rel:
            rel_cnt[rel] += 1
        src = r.get("source", "")
        if src:
            src_cnt[src] += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("三元组数", f"{len(rows):,}")
    c2.metric("实体数", f"{len(ents):,}")
    c3.metric("关系类型数", f"{len(rel_cnt):,}")

    with st.expander("Top-20 关系分布", expanded=False):
        for rel, n in rel_cnt.most_common(20):
            st.write(f"- `{rel}` × {n}")

    if src_cnt:
        with st.expander("按来源分布", expanded=False):
            for s, n in src_cnt.most_common(20):
                st.write(f"- `{s}` × {n}")


def main() -> None:
    st.set_page_config(
        page_title="单-双折叠翼变体飞行器 · 知识图谱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("单-双折叠翼变体飞行器 · 知识图谱浏览")
    st.caption("基于清华大学郭廷宇博士论文，采用 NER + 触发词 + 模板 + SVO + 类型 + 章节 多策略抽取")

    with st.sidebar:
        st.header("数据源")
        path = st.text_input("三元组 CSV", value=_DEFAULT_CSV)

        st.header("筛选")
        min_score = st.slider("最小 score", 0.0, 1.0, 0.40, 0.05)
        min_text_len = st.slider("最小实体长度", 1, 6, 2, 1)
        max_nodes = st.slider("子图最大节点数", 10, 300, 80, 10)
        max_hops = st.slider("BFS 最大跳数", 1, 5, 2, 1)

    if not os.path.isfile(path):
        st.error(f"找不到文件：`{path}`。请先运行 `uv run python run_extract.py`")
        return

    rows = load_triples(path)
    ent_types = load_entity_types(_ENTITY_TYPE_JSON)
    _stats_panel(rows)

    all_relations = sorted({_row_get(r, "relation", "关系") for r in rows if _row_get(r, "relation", "关系")})
    with st.sidebar:
        st.header("关系白名单")
        sel = st.multiselect(
            "勾选要展示的关系（不选=全选）",
            options=all_relations,
            default=[],
        )

    rel_whitelist = set(sel) if sel else None

    entities = sorted({
        e
        for r in rows
        for e in (_row_get(r, "head", "头实体"), _row_get(r, "tail", "尾实体"))
        if e
    })
    if not entities:
        st.warning("解析不到实体，请确认 CSV 含 head/tail 列。")
        return

    # 推荐高连通度实体作为默认中心
    deg: Counter[str] = Counter()
    for r in rows:
        h = _row_get(r, "head", "头实体")
        t = _row_get(r, "tail", "尾实体")
        deg[h] += 1
        deg[t] += 1
    default_center = deg.most_common(1)[0][0] if deg else entities[0]
    center = st.sidebar.selectbox(
        "中心实体",
        options=entities,
        index=entities.index(default_center) if default_center in entities else 0,
    )

    st.divider()
    st.subheader(f"图谱子图 · 中心={center}")
    g = build_subgraph(
        rows, center,
        max_nodes=max_nodes,
        min_score=min_score,
        rel_whitelist=rel_whitelist,
        min_text_len=min_text_len,
        max_hops=max_hops,
    )
    if g.number_of_nodes() == 0:
        st.warning("当前过滤下该中心实体无邻接边，请降低 score 或换实体。")
        return

    html = to_pyvis_html(g, ent_types, center)
    st.components.v1.html(html, height=640, scrolling=False)
    st.caption(f"子图节点数 {g.number_of_nodes()}，边数 {g.number_of_edges()}")

    with st.expander("当前子图边明细", expanded=False):
        for u, v, _k, data in g.edges(keys=True, data=True):
            st.write(
                f"- **{u}** — `{data.get('label', '')}` → **{v}** "
                f"(score={data.get('score', 1.0):.2f}, source={data.get('source', '?')})"
            )


if __name__ == "__main__":
    main()
