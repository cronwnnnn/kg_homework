"""单-双折叠翼变体飞行器 · 领域问答助手 (QA + 图谱双面板)。

模块组成：
    kg_store       : 知识图谱内存视图（networkx + 倒排索引）。
    corpus_index   : 论文原文句子索引 (实体 → 句子位置)。
    query_parser   : 自然语言问题解析（类型 + 实体）。
    retriever      : 图谱与原文协同检索。
    answer_builder : 模板答案合成 + 可选 LLM 增强。
    llm_client     : OpenAI 兼容接口，未配置时静默降级到 mock。
    app            : Streamlit 双面板主 UI 入口。
"""

__all__ = [
    "kg_store",
    "corpus_index",
    "query_parser",
    "retriever",
    "answer_builder",
    "llm_client",
]
