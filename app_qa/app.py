"""单-双折叠翼变体飞行器 · 领域问答助手 (Streamlit 双面板)。

左侧：自然语言问答（问题输入 → 解析 → 答案 → 三元组依据 → 原文证据）
右侧：知识图谱可视化（中心实体子图 + 高亮主答案路径）

运行：
    uv run streamlit run app_qa/app.py
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from collections import Counter

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import networkx as nx  # noqa: E402
from pyvis.network import Network  # noqa: E402

from app_qa.answer_builder import build_answer  # noqa: E402
from app_qa.corpus_index import Corpus  # noqa: E402
from app_qa.kg_store import KGStore, TripleRow, load_triples  # noqa: E402
from app_qa.llm_client import LLMClient  # noqa: E402
from app_qa.query_parser import ParsedQuery, QueryParser  # noqa: E402
from app_qa.retriever import retrieve  # noqa: E402


_DEFAULT_TRIPLES = os.path.join(ROOT, "output", "triples_with_meta.csv")
_DEFAULT_CORPUS = os.path.join(ROOT, "aftcln.txt")
_ENTITY_TYPE_JSON = os.path.join(ROOT, "data", "entities_by_type.json")
_EXAMPLE_FILE = os.path.join(ROOT, "app_qa", "samples", "example_questions.txt")


def _resolve_path(p: str) -> str:
    """允许用户在 sidebar 输入相对路径：自动按项目根解析。"""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    cand = os.path.join(ROOT, p)
    return cand if os.path.exists(cand) else p


_TYPE_COLOR: dict[str, str] = {
    "AIRCRAFT":             "#4a90e2",
    "WING_CONFIGURATION":   "#48b884",
    "PARAMETER":            "#e8b339",
    "AERODYNAMIC_CONCEPT":  "#ec8064",
    "STRUCTURAL_COMPONENT": "#9b7bd4",
    "CONTROL_METHOD":       "#e36a7e",
    "PERFORMANCE_METRIC":   "#3eafa8",
    "ORGANIZATION":         "#7a8fa6",
    "PERSON":               "#e6925e",
    "TECHNOLOGY":           "#5b8fd6",
    "MATERIAL":             "#5ba994",
    "FLIGHT_PHASE":         "#d18752",
    "EQUATION":             "#9aa5b1",
    "CONCEPT":              "#c97485",
    "_UNKNOWN_":            "#b8c1cc",
}


_TYPE_LABEL_ZH: dict[str, str] = {
    "AIRCRAFT": "飞行器",
    "WING_CONFIGURATION": "机翼构型",
    "PARAMETER": "参数",
    "AERODYNAMIC_CONCEPT": "气动概念",
    "STRUCTURAL_COMPONENT": "结构部件",
    "CONTROL_METHOD": "控制方法",
    "PERFORMANCE_METRIC": "性能指标",
    "ORGANIZATION": "组织",
    "PERSON": "人物",
    "TECHNOLOGY": "技术",
    "MATERIAL": "材料",
    "FLIGHT_PHASE": "飞行阶段",
    "EQUATION": "方程/公式",
    "CONCEPT": "概念",
    "_UNKNOWN_": "未识别",
}


_REL_GROUP: dict[str, tuple[str, str, str]] = {
    "instance_of":          ("分类层级", "#5b6ee1", "solid"),
    "is_a":                 ("分类层级", "#5b6ee1", "solid"),

    "has_part":             ("组成结构", "#4a89dc", "solid"),
    "part_of":              ("组成结构", "#4a89dc", "solid"),
    "connected_to":         ("组成结构", "#4a89dc", "solid"),
    "combines_with":        ("组成结构", "#4a89dc", "solid"),

    "causes":               ("因果影响", "#e8804b", "solid"),
    "leads_to":             ("因果影响", "#e8804b", "solid"),
    "affects":              ("因果影响", "#e8804b", "dashed"),
    "generates":            ("因果影响", "#e8804b", "solid"),
    "provides":             ("因果影响", "#e8804b", "solid"),

    "improves":             ("性能改善", "#37b87f", "solid"),
    "enhances":             ("性能改善", "#37b87f", "solid"),
    "satisfies":            ("性能改善", "#37b87f", "solid"),
    "verifies":             ("性能改善", "#37b87f", "dashed"),

    "reduces":              ("性能下降", "#e85d75", "solid"),
    "solves":               ("性能下降", "#e85d75", "solid"),

    "has_value":            ("数值比较", "#c2a663", "solid"),
    "greater_than":         ("数值比较", "#c2a663", "solid"),
    "less_than":            ("数值比较", "#c2a663", "solid"),
    "equals_to":            ("数值比较", "#c2a663", "solid"),
    "greater_than_value":   ("数值比较", "#c2a663", "solid"),
    "less_than_value":      ("数值比较", "#c2a663", "solid"),
    "approximately":        ("数值比较", "#c2a663", "dashed"),

    "controls":             ("控制驱动", "#3eafa8", "solid"),
    "controlled_by":        ("控制驱动", "#3eafa8", "solid"),
    "drives":               ("控制驱动", "#3eafa8", "solid"),
    "driven_by":            ("控制驱动", "#3eafa8", "solid"),
    "actuated_by":          ("控制驱动", "#3eafa8", "solid"),

    "used_for":             ("使用应用", "#7a8fa6", "solid"),
    "uses_method":          ("使用应用", "#7a8fa6", "solid"),
    "applied_to":           ("使用应用", "#7a8fa6", "solid"),
    "implements":           ("使用应用", "#7a8fa6", "solid"),
    "needs":                ("使用应用", "#7a8fa6", "dashed"),
    "depends_on":           ("使用应用", "#7a8fa6", "dashed"),
    "enables":              ("使用应用", "#7a8fa6", "solid"),

    "develops":             ("开发研制", "#9166cc", "solid"),
    "developed_by":         ("开发研制", "#9166cc", "solid"),
    "manufactures":         ("开发研制", "#9166cc", "solid"),
    "originates_from":      ("开发研制", "#9166cc", "dashed"),

    "located_at":           ("位置变换", "#b08968", "solid"),
    "contains":             ("位置变换", "#b08968", "solid"),
    "transforms_to":        ("位置变换", "#b08968", "solid"),
    "transformed_from":     ("位置变换", "#b08968", "solid"),
    "generated_by":         ("位置变换", "#b08968", "solid"),

    "discussed_in":         ("章节归属", "#a0a8b3", "dashed"),
    "co_occurs_with":       ("章节归属", "#a0a8b3", "dashed"),
}


_REL_DEFAULT = ("其他", "#9aa5b1", "solid")


def _rel_style(relation: str) -> tuple[str, str, str]:
    return _REL_GROUP.get(relation, _REL_DEFAULT)


_THEME = {
    "bg":          "#ffffff",
    "panel_bg":    "#f7f9fc",
    "node_text":   "#1f2933",
    "edge_text":   "#5a6573",
    "muted":       "#8a96a3",
    "center_ring": "#ff9f43",
    "center_text": "#3d2c00",
    "highlight":   "#ffb84d",
    "node_border": "#d6dde6",
    "edge_default":"#cbd2dc",
}


# ---------- cache ----------

@st.cache_data(show_spinner=False)
def _cached_triples(path: str) -> list[TripleRow]:
    return load_triples(path)


@st.cache_resource(show_spinner=False)
def _cached_kg(path: str) -> KGStore:
    return KGStore(_cached_triples(path))


@st.cache_resource(show_spinner=False)
def _cached_corpus(path: str) -> Corpus:
    return Corpus(path)


@st.cache_resource(show_spinner=False)
def _cached_parser(vocab_tuple: tuple[str, ...]) -> QueryParser:
    return QueryParser(vocab=vocab_tuple)


@st.cache_data(show_spinner=False)
def _cached_entity_types(path: str) -> dict[str, str]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
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


@st.cache_data(show_spinner=False)
def _cached_examples(path: str) -> list[str]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8-sig") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


# ---------- pyvis ----------

def _build_visual_graph(
    kg: KGStore,
    seed: str,
    highlight_triples: list[TripleRow],
    hops: int,
    max_nodes: int,
) -> nx.MultiDiGraph:
    g = kg.neighbors_subgraph(seed, hops=hops, max_nodes=max_nodes)
    # 把高亮三元组的两端都纳入图（即使它们超出 hops，也优先展示主答案）
    for t in highlight_triples:
        if t.head not in g:
            g.add_node(t.head)
        if t.tail not in g:
            g.add_node(t.tail)
        g.add_edge(t.head, t.tail, label=t.relation, score=t.score, source=t.source, _hl=True)
    return g


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(180,180,180,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def to_pyvis_html(
    g: nx.MultiDiGraph,
    ent_types: dict[str, str],
    center: str,
    highlight_pairs: set[tuple[str, str]],
) -> str:
    net = Network(
        height="660px",
        width="100%",
        directed=True,
        bgcolor=_THEME["bg"],
        font_color=_THEME["node_text"],
    )
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": false,
        "zoomView": true,
        "dragNodes": true
      },
      "physics": {
        "enabled": true,
        "stabilization": { "enabled": true, "iterations": 220, "fit": true },
        "barnesHut": {
          "gravitationalConstant": -9000,
          "centralGravity": 0.18,
          "springLength": 160,
          "springConstant": 0.035,
          "damping": 0.55,
          "avoidOverlap": 0.6
        },
        "minVelocity": 0.6,
        "timestep": 0.45
      },
      "nodes": {
        "shape": "dot",
        "borderWidth": 1.5,
        "borderWidthSelected": 3,
        "shadow": {
          "enabled": true,
          "color": "rgba(70,90,120,0.18)",
          "size": 12,
          "x": 0,
          "y": 4
        },
        "font": {
          "face": "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif",
          "color": "#1f2933",
          "strokeWidth": 4,
          "strokeColor": "rgba(255,255,255,0.95)"
        }
      },
      "edges": {
        "smooth": {
          "enabled": true,
          "type": "continuous",
          "roundness": 0.32
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.55, "type": "arrow" }
        },
        "font": {
          "face": "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif",
          "color": "#5a6573",
          "size": 11,
          "strokeWidth": 3,
          "strokeColor": "rgba(255,255,255,0.95)",
          "align": "middle"
        },
        "selectionWidth": 1.4,
        "hoverWidth": 1.2
      }
    }
    """)

    for n in g.nodes():
        t = ent_types.get(n) or "_UNKNOWN_"
        color = _TYPE_COLOR.get(t, _TYPE_COLOR["_UNKNOWN_"])
        is_center = (n == center)
        is_hl = (not is_center) and any(n == a or n == b for a, b in highlight_pairs)

        bg = color
        hover_bg = color
        border = _THEME["node_border"]
        font_color = _THEME["node_text"]
        size = 18
        font_size = 13

        if is_center:
            bg = color
            border = _THEME["center_ring"]
            font_color = _THEME["center_text"]
            size = 34
            font_size = 17
        elif is_hl:
            border = _THEME["highlight"]
            size = 24
            font_size = 14

        type_zh = _TYPE_LABEL_ZH.get(t, t)
        deg = g.degree(n)
        tooltip = (
            f"<div style='font-family:Inter,sans-serif;'>"
            f"<div style='font-size:14px;font-weight:600;color:#1f2933'>{n}</div>"
            f"<div style='font-size:12px;color:#5a6573;margin-top:2px'>类型: {type_zh}</div>"
            f"<div style='font-size:12px;color:#8a96a3;margin-top:2px'>邻接度: {deg}</div>"
            f"</div>"
        )

        net.add_node(
            n,
            label=n,
            title=tooltip,
            color={
                "background": bg,
                "border": border,
                "highlight": {"background": hover_bg, "border": _THEME["center_ring"]},
                "hover": {"background": hover_bg, "border": _THEME["highlight"]},
            },
            size=size,
            borderWidth=3.5 if is_center else (2.5 if is_hl else 1.5),
            font={
                "size": font_size,
                "color": font_color,
                "face": "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif",
                "strokeWidth": 4,
                "strokeColor": "rgba(255,255,255,0.95)",
            },
        )

    edge_seen: dict[tuple[str, str, str], int] = {}
    for u, v, _k, data in g.edges(keys=True, data=True):
        lab = data.get("label", "")
        sc = data.get("score", 1.0)
        src = data.get("source", "")
        is_hl = bool(data.get("_hl")) or ((u, v) in highlight_pairs) or ((v, u) in highlight_pairs)

        group_name, group_color, dash = _rel_style(lab)
        edge_color = _THEME["highlight"] if is_hl else group_color
        edge_color_rgba = edge_color if is_hl else _hex_to_rgba(edge_color, 0.78)

        if is_hl:
            width = 3.2
        elif sc >= 0.85:
            width = 2.0
        elif sc >= 0.65:
            width = 1.5
        else:
            width = 1.0

        pair_key = (u, v, lab)
        edge_seen[pair_key] = edge_seen.get(pair_key, 0) + 1
        smooth_type = "curvedCW" if edge_seen[pair_key] % 2 else "curvedCCW"
        roundness = 0.18 + 0.12 * (edge_seen[pair_key] - 1)
        if roundness > 0.7:
            roundness = 0.7

        tooltip_edge = (
            f"<div style='font-family:Inter,sans-serif;'>"
            f"<div style='font-size:13px;color:#1f2933'>"
            f"<span style='font-weight:600'>{u}</span>"
            f" <span style='color:{group_color};font-weight:600'>{lab}</span> "
            f"<span style='font-weight:600'>{v}</span></div>"
            f"<div style='font-size:11px;color:#5a6573;margin-top:3px'>"
            f"语义组: {group_name}　·　score={sc:.2f}　·　source={src or '-'}"
            f"</div></div>"
        )

        net.add_edge(
            u, v,
            title=tooltip_edge,
            label=lab[:14] + ("…" if len(lab) > 14 else ""),
            color={"color": edge_color_rgba, "highlight": _THEME["highlight"], "hover": _THEME["highlight"]},
            width=width,
            dashes=(dash == "dashed"),
            smooth={"enabled": True, "type": smooth_type, "roundness": roundness},
            font={
                "size": 11 if not is_hl else 12,
                "color": _THEME["edge_text"] if not is_hl else "#b46300",
                "face": "Inter, 'PingFang SC', 'Microsoft YaHei', sans-serif",
                "strokeWidth": 3,
                "strokeColor": "rgba(255,255,255,0.95)",
                "align": "middle",
            },
            arrows={"to": {"enabled": True, "scaleFactor": 0.55, "type": "arrow"}},
        )
    try:
        return net.generate_html(notebook=False)
    except TypeError:
        try:
            return net.generate_html()
        except Exception:
            return getattr(net, "html", "") or "<p>pyvis 渲染失败</p>"


# ---------- panels ----------

def _render_left_panel(
    question: str,
    parser: QueryParser,
    kg: KGStore,
    corpus: Corpus,
    llm: LLMClient,
    use_llm: bool,
) -> tuple[ParsedQuery, list[TripleRow], str]:
    parsed = parser.parse(question)
    st.markdown("### 解析结果")
    st.markdown(
        f"- 意图: `{parsed.intent.value}`"
        + (f"  · 目标关系提示: `{parsed.target_relation}`" if parsed.target_relation else "")
    )
    if parsed.entities:
        chips = " ".join(f"`{e}`" for e in parsed.entities)
        st.markdown(f"- 识别到的实体: {chips}")
    else:
        st.warning("未识别到任何领域实体，回答可能不准确。")

    retr = retrieve(parsed, kg, corpus)
    if use_llm and llm.available:
        with st.spinner("LLM 正在重写回答…"):
            answer = build_answer(parsed, retr, llm=llm, use_llm=use_llm)
    else:
        answer = build_answer(parsed, retr, llm=llm, use_llm=use_llm)
    if llm.last_error:
        st.session_state["llm_last_error"] = llm.last_error
    elif answer.llm_used:
        st.session_state.pop("llm_last_error", None)

    st.markdown("### 回答")
    if answer.llm_used:
        st.caption("(由 LLM 基于图谱+原文重写)")
    else:
        st.caption("(模板合成；可在侧栏开启 LLM 重写以获得更自然的回答)")
    st.markdown(answer.text)

    if answer.notes:
        for note in answer.notes:
            st.info(note)

    if answer.used_triples:
        st.markdown("### 主答案三元组")
        for t in answer.used_triples:
            st.markdown(
                f"- **{t.head}** — `{t.relation}` → **{t.tail}** "
                f"<span style='color:#888'>(score={t.score:.2f}, source={t.source or '?'})</span>",
                unsafe_allow_html=True,
            )

    if retr.related_triples:
        with st.expander("更多相关三元组", expanded=False):
            for t in retr.related_triples:
                st.markdown(
                    f"- {t.head} — `{t.relation}` → {t.tail} "
                    f"<span style='color:#777'>({t.source})</span>",
                    unsafe_allow_html=True,
                )

    if answer.evidences:
        st.markdown("### 原文证据")
        for ev in answer.evidences[:8]:
            head = (ev.section or ev.chapter or "正文").strip()
            st.markdown(
                f"<div style='padding:6px 10px;margin:4px 0;background:#1c2128;"
                f"border-left:3px solid #4a9eff;border-radius:4px'>"
                f"<span style='color:#7ab7ff'>[{head}]</span> {ev.text}</div>",
                unsafe_allow_html=True,
            )

    return parsed, answer.used_triples, parsed.primary or ""


def _render_right_panel(
    kg: KGStore,
    ent_types: dict[str, str],
    seed: str,
    highlight: list[TripleRow],
    hops: int,
    max_nodes: int,
) -> None:
    st.markdown("### 知识图谱可视化")
    if not seed:
        st.info("请先在左侧提一个问题，可视化将以问题中的实体为中心展开。")
        return
    if seed not in kg.graph:
        st.warning(f"实体「{seed}」不在图谱中，无法绘制子图。")
        candidates = difflib.get_close_matches(seed, kg.all_entities(), n=6, cutoff=0.4)
        if candidates:
            st.caption("您是否想问以下相近实体？点击即可作为中心绘制：")
            cols = st.columns(min(len(candidates), 3))
            for i, c in enumerate(candidates):
                if cols[i % len(cols)].button(c, key=f"sim_{c}", use_container_width=True):
                    st.session_state["q_input"] = c
                    st.rerun()
        return

    g = _build_visual_graph(kg, seed, highlight, hops, max_nodes)
    hl_pairs = {(t.head, t.tail) for t in highlight}
    html = to_pyvis_html(g, ent_types, seed, hl_pairs)

    st.markdown(
        "<div style='border:1px solid #e4e9f0;border-radius:10px;background:#ffffff;"
        "padding:6px;box-shadow:0 1px 4px rgba(70,90,120,0.06);margin-top:4px'>",
        unsafe_allow_html=True,
    )
    st.components.v1.html(html, height=680, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metric_cols[0].markdown(f"<div style='font-size:12px;color:#8a96a3'>中心实体</div>"
                            f"<div style='font-size:14px;font-weight:600;color:#1f2933'>{seed}</div>",
                            unsafe_allow_html=True)
    metric_cols[1].markdown(f"<div style='font-size:12px;color:#8a96a3'>BFS 跳数</div>"
                            f"<div style='font-size:14px;font-weight:600;color:#1f2933'>{hops}</div>",
                            unsafe_allow_html=True)
    metric_cols[2].markdown(f"<div style='font-size:12px;color:#8a96a3'>节点 / 边数</div>"
                            f"<div style='font-size:14px;font-weight:600;color:#1f2933'>"
                            f"{g.number_of_nodes()} / {g.number_of_edges()}</div>",
                            unsafe_allow_html=True)
    metric_cols[3].markdown(f"<div style='font-size:12px;color:#8a96a3'>高亮主答案边</div>"
                            f"<div style='font-size:14px;font-weight:600;color:#1f2933'>{len(highlight)}</div>",
                            unsafe_allow_html=True)

    type_set = sorted({ent_types.get(n) or "_UNKNOWN_" for n in g.nodes()})
    rel_set: dict[str, tuple[str, str]] = {}
    for _u, _v, _k, ed in g.edges(keys=True, data=True):
        rel = ed.get("label", "")
        if not rel:
            continue
        group_name, group_color, _ = _rel_style(rel)
        rel_set.setdefault(group_name, (group_color, rel))

    with st.expander("图例 · 节点类型与关系语义", expanded=False):
        st.markdown("<div style='font-size:12px;color:#5a6573;margin-bottom:6px'><b>节点类型</b></div>",
                    unsafe_allow_html=True)
        cols = st.columns(4)
        for i, tname in enumerate(type_set):
            c = _TYPE_COLOR.get(tname, _TYPE_COLOR["_UNKNOWN_"])
            display = _TYPE_LABEL_ZH.get(tname, tname)
            cols[i % 4].markdown(
                f"<div style='display:flex;align-items:center;margin:3px 0;'>"
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{c};border-radius:50%;margin-right:8px;"
                f"box-shadow:0 1px 2px rgba(70,90,120,0.18)'></span>"
                f"<span style='font-size:12px;color:#1f2933'>{display}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if rel_set:
            st.markdown("<div style='font-size:12px;color:#5a6573;margin:10px 0 6px'>"
                        "<b>关系语义组</b></div>",
                        unsafe_allow_html=True)
            cols2 = st.columns(3)
            for i, (group_name, (group_color, sample_rel)) in enumerate(sorted(rel_set.items())):
                cols2[i % 3].markdown(
                    f"<div style='display:flex;align-items:center;margin:3px 0;'>"
                    f"<span style='display:inline-block;width:22px;height:3px;"
                    f"background:{group_color};border-radius:2px;margin-right:8px;'></span>"
                    f"<span style='font-size:12px;color:#1f2933'>{group_name}</span>"
                    f"<span style='font-size:11px;color:#8a96a3;margin-left:6px'>(如 {sample_rel})</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _render_stats(kg: KGStore) -> None:
    rel_cnt: Counter[str] = Counter()
    src_cnt: Counter[str] = Counter()
    for t in kg.iter_triples():
        rel_cnt[t.relation] += 1
        if t.source:
            src_cnt[t.source] += 1
    c1, c2, c3 = st.columns(3)
    c1.metric("实体数", f"{len(kg.all_entities()):,}")
    c2.metric("三元组数", f"{len(kg):,}")
    c3.metric("关系类型数", f"{len(rel_cnt):,}")
    with st.expander("Top-15 关系", expanded=False):
        for rel, n in rel_cnt.most_common(15):
            st.write(f"- `{rel}` × {n}")
    if src_cnt:
        with st.expander("Top-10 来源", expanded=False):
            for s, n in src_cnt.most_common(10):
                st.write(f"- `{s}` × {n}")


# ---------- main ----------

_GLOBAL_CSS = """
<style>
  .stApp { background: linear-gradient(180deg, #fbfcfe 0%, #f4f7fb 100%) !important; }
  section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e4e9f0; }
  h1, h2, h3 { color: #1f2933 !important; font-family: Inter, 'PingFang SC', sans-serif; }
  .stMarkdown, .stCaption { color: #3d4852; }
  div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e4e9f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(70,90,120,0.05) !important;
  }
  div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e4e9f0;
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 1px 3px rgba(70,90,120,0.06);
  }
  .stButton button {
    border-radius: 8px !important;
    border: 1px solid #d6dde6 !important;
    background: #ffffff !important;
    color: #1f2933 !important;
    transition: all .15s ease;
  }
  .stButton button:hover {
    border-color: #4a90e2 !important;
    color: #4a90e2 !important;
    background: #f0f6ff !important;
  }
  input[type="text"] {
    border-radius: 8px !important;
    border: 1px solid #d6dde6 !important;
  }
  input[type="text"]:focus {
    border-color: #4a90e2 !important;
    box-shadow: 0 0 0 3px rgba(74,144,226,0.12) !important;
  }
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="单-双折叠翼变体飞行器 · 领域问答助手",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div style='padding:14px 18px;border-radius:12px;"
        "background:linear-gradient(135deg,#ffffff,#f0f6ff);"
        "border:1px solid #e4e9f0;box-shadow:0 1px 3px rgba(70,90,120,0.06);"
        "margin-bottom:10px'>"
        "<div style='font-size:24px;font-weight:700;color:#1f2933;font-family:Inter,sans-serif'>"
        "✈ 单-双折叠翼变体飞行器 · 领域问答助手"
        "</div>"
        "<div style='font-size:13px;color:#5a6573;margin-top:4px'>"
        "基于知识图谱（NER + 触发词 + 模板 + 依存增强 + 类型/章节）与论文原文的双面板 QA 系统"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("数据源")
        triples_path = _resolve_path(st.text_input("三元组 CSV", value=_DEFAULT_TRIPLES))
        corpus_path = _resolve_path(st.text_input("论文文本", value=_DEFAULT_CORPUS))

        st.header("可视化参数")
        hops = st.slider("BFS 跳数", 1, 4, 2)
        max_nodes = st.slider("子图最大节点数", 10, 200, 60, 5)

        st.header("LLM 增强")
        use_llm = st.checkbox("启用 LLM 重写回答", value=False, help="勾选后将以模板答案+图谱+原文为材料调 LLM 重写更自然的中文回答；未配置 OPENAI_API_KEY 时静默降级回模板。")

        with st.expander("LLM 配置（可选）", expanded=False):
            api_key = st.text_input("OPENAI_API_KEY", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
            base_url = st.text_input("OPENAI_BASE_URL", value=os.environ.get("OPENAI_BASE_URL", ""), placeholder="可填 DeepSeek/Kimi 等兼容端点")
            model_name = st.text_input("Model", value=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))

    if not os.path.isfile(triples_path):
        st.error(f"找不到三元组 CSV: `{triples_path}`，请先运行 `uv run python run_extract.py`")
        return
    if not os.path.isfile(corpus_path):
        st.error(f"找不到论文文件: `{corpus_path}`")
        return

    with st.spinner("加载知识图谱与论文…"):
        kg = _cached_kg(triples_path)
        corpus = _cached_corpus(corpus_path)
        ent_types = _cached_entity_types(_ENTITY_TYPE_JSON)
        vocab = tuple(sorted(kg.all_entities()))
        parser = _cached_parser(vocab)

    llm = LLMClient(api_key=api_key or None, base_url=base_url or None, model=model_name or None)

    with st.sidebar:
        st.header("LLM 状态")
        if llm.available:
            st.success(f"LLM 已就绪 · model={llm.model}")
        else:
            st.info("LLM 未配置 · 仅用模板答案")
        last_err = st.session_state.get("llm_last_error")
        if last_err:
            st.warning(f"上次 LLM 错误：{last_err}")

    _render_stats(kg)
    st.divider()

    # 例题
    examples = _cached_examples(_EXAMPLE_FILE)
    if examples:
        with st.expander("示例问题（点击复用）", expanded=False):
            for i, ex in enumerate(examples):
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state["q_input"] = ex

    if "q_input" not in st.session_state:
        st.session_state["q_input"] = ""
    question = st.text_input(
        "请输入您的问题",
        key="q_input",
        placeholder="例如：什么是变体飞行器？  ／  全球鹰是多少 ／ 折叠翼会影响什么 ／ 升力 和 阻力 的关系",
    )

    if not question:
        st.info("请输入问题并按回车键发送 ↑")
        return

    col_left, col_right = st.columns([1.0, 1.2], gap="large")
    with col_left:
        parsed, highlight_triples, seed = _render_left_panel(
            question, parser, kg, corpus, llm, use_llm,
        )
    with col_right:
        _render_right_panel(kg, ent_types, seed, highlight_triples, hops, max_nodes)


if __name__ == "__main__":
    main()
