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


def to_pyvis_html(
    g: nx.MultiDiGraph,
    ent_types: dict[str, str],
    center: str,
    highlight_pairs: set[tuple[str, str]],
) -> str:
    net = Network(
        height="640px",
        width="100%",
        directed=True,
        bgcolor="#0f1419",
        font_color="#eeeeee",
    )
    net.barnes_hut(gravity=-7000, central_gravity=0.3, spring_length=130, spring_strength=0.05)

    for n in g.nodes():
        t = ent_types.get(n) or "_UNKNOWN_"
        color = _TYPE_COLOR.get(t, _TYPE_COLOR["_UNKNOWN_"])
        is_center = (n == center)
        is_hl = any(n == a or n == b for a, b in highlight_pairs)
        size = 32 if is_center else (22 if is_hl else 16)
        border = "#ffd54a" if is_center else ("#ff6b6b" if is_hl else "#444")
        net.add_node(
            n,
            label=n,
            title=f"{n}\n类型: {t if t != '_UNKNOWN_' else '未识别'}",
            color={"background": color, "border": border},
            size=size,
            borderWidth=3 if (is_center or is_hl) else 1,
            font={"size": 17 if is_center else (14 if is_hl else 12), "color": "#eee"},
        )

    for u, v, _k, data in g.edges(keys=True, data=True):
        lab = data.get("label", "")
        sc = data.get("score", 1.0)
        src = data.get("source", "")
        is_hl = data.get("_hl") or ((u, v) in highlight_pairs) or ((v, u) in highlight_pairs)
        title = f"{u} --[{lab}]--> {v}\nscore={sc:.2f}  source={src}"
        net.add_edge(
            u, v,
            title=title,
            label=lab[:18] + ("…" if len(lab) > 18 else ""),
            color="#ffd54a" if is_hl else "#888",
            width=3 if is_hl else 1,
            font={"size": 11, "color": "#cccccc"},
            arrows={"to": {"enabled": True}},
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
    st.components.v1.html(html, height=660, scrolling=False)
    st.caption(
        f"中心: **{seed}** | BFS 跳数: {hops} | 节点数: {g.number_of_nodes()} | "
        f"边数: {g.number_of_edges()} | 高亮主答案边: {len(highlight)}"
    )

    type_set = sorted({ent_types.get(n) or "_UNKNOWN_" for n in g.nodes()})
    with st.expander("图例 · 颜色 / 类型", expanded=False):
        cols = st.columns(4)
        for i, tname in enumerate(type_set):
            c = _TYPE_COLOR.get(tname, _TYPE_COLOR["_UNKNOWN_"])
            display = tname if tname != "_UNKNOWN_" else "未识别"
            cols[i % 4].markdown(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{c};border-radius:50%;margin-right:6px'></span>{display}",
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

def main() -> None:
    st.set_page_config(
        page_title="单-双折叠翼变体飞行器 · 领域问答助手",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("✈ 单-双折叠翼变体飞行器 · 领域问答助手")
    st.caption("基于知识图谱 (NER+触发词+模板+SVO+类型/章节) 与论文原文的双面板 QA 系统")

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
