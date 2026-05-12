"""答案合成：基于 RetrievalResult + 原文证据，使用模板生成回答；
可选 LLM 总结增强（失败/未配置则降级到模板）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus_index import SentenceRecord
from .kg_store import TripleRow
from .llm_client import LLMClient
from .query_parser import ParsedQuery, QueryIntent
from .retriever import RetrievalResult


@dataclass
class Answer:
    text: str = ""
    bullets: list[str] = field(default_factory=list)
    used_triples: list[TripleRow] = field(default_factory=list)
    evidences: list[SentenceRecord] = field(default_factory=list)
    intent: QueryIntent = QueryIntent.UNKNOWN
    llm_used: bool = False
    notes: list[str] = field(default_factory=list)


def _fmt_triple(t: TripleRow) -> str:
    return f"({t.head} —{t.relation}→ {t.tail})"


def _fmt_evidence(ev: SentenceRecord) -> str:
    head = (ev.section or ev.chapter or "正文").strip()
    return f"[{head}] {ev.text}"


def _intent_lead(intent: QueryIntent, primary: str, secondary: str) -> str:
    mapping = {
        QueryIntent.DEFINITION: f"关于「{primary}」是什么，知识图谱中给出的回答如下：",
        QueryIntent.LIST_INSTANCE: f"图谱中已收录的「{primary}」类相关实体如下：",
        QueryIntent.VALUE: f"「{primary}」的相关数值/取值信息：",
        QueryIntent.CAUSE: f"导致 / 影响「{primary}」的因素：",
        QueryIntent.EFFECT: f"「{primary}」会带来 / 影响的结果：",
        QueryIntent.LOCATION: f"「{primary}」所处的位置 / 部位信息：",
        QueryIntent.CHAPTER: f"「{primary}」在论文中被讨论的章节：",
        QueryIntent.PROPERTY: f"「{primary}」的属性 / 参数信息：",
        QueryIntent.RELATION: f"「{primary}」与「{secondary}」之间的关系：",
        QueryIntent.SUMMARY: f"图谱中关于「{primary}」的关键信息汇总：",
        QueryIntent.NEIGHBOR: f"「{primary}」的相关三元组：",
        QueryIntent.UNKNOWN: f"图谱中与「{primary}」相关的信息：",
    }
    return mapping.get(intent, mapping[QueryIntent.UNKNOWN])


def _template_compose(parsed: ParsedQuery, retr: RetrievalResult) -> Answer:
    ans = Answer(intent=parsed.intent, used_triples=retr.triples, evidences=retr.evidence_rows)
    if retr.note:
        ans.notes.append(retr.note)

    if not retr.triples and not retr.evidence_rows:
        ans.text = (
            f"未在知识图谱与论文中找到与「{parsed.primary or parsed.raw}」直接相关的内容。"
            "请尝试更换关键词，或在左侧切换为「全局浏览」模式查看可用实体。"
        )
        return ans

    lead = _intent_lead(parsed.intent, parsed.primary, parsed.secondary)
    bullets = [_fmt_triple(t) for t in retr.triples[:12]]
    ans.bullets = bullets
    body_lines = [lead]
    if bullets:
        body_lines.extend(f"  • {b}" for b in bullets)
    else:
        body_lines.append("  • （图谱中暂无直接三元组，请参考下方原文证据）")
    if retr.evidence_rows:
        body_lines.append("")
        body_lines.append("原文佐证：")
        for ev in retr.evidence_rows[:3]:
            body_lines.append(f"  · {_fmt_evidence(ev)}")
    ans.text = "\n".join(body_lines)
    return ans


_LLM_PROMPT = """你是航空知识问答助手。请根据给定的「知识图谱三元组」与「论文原文证据」，对用户问题给出准确、简洁的中文回答。

要求：
1) 回答先用一两句话直接给出结论，再分点列出依据。
2) 仅使用「材料」中出现的信息；如材料不足，请明确说"图谱与论文未提供更多信息"。
3) 引用三元组时使用 (头实体 — 关系 → 尾实体) 的格式。
4) 不要编造，不要发挥与材料无关的内容。

用户问题：
{question}

意图：{intent}
主要实体：{primary}
次要实体：{secondary}

知识图谱三元组：
{triples}

论文原文证据：
{evidence}

请给出回答："""


def _build_llm_prompt(parsed: ParsedQuery, retr: RetrievalResult) -> str:
    triple_lines = [
        f"- ({t.head} | {t.relation} | {t.tail})  [source={t.source}, score={t.score:.2f}]"
        for t in retr.triples[:20]
    ] or ["- （无）"]
    evidence_lines = [
        f"- [{(ev.section or ev.chapter or '正文').strip()}] {ev.text}"
        for ev in retr.evidence_rows[:6]
    ] or ["- （无）"]
    return _LLM_PROMPT.format(
        question=parsed.raw,
        intent=parsed.intent.value,
        primary=parsed.primary or "（未识别）",
        secondary=parsed.secondary or "（无）",
        triples="\n".join(triple_lines),
        evidence="\n".join(evidence_lines),
    )


def build_answer(
    parsed: ParsedQuery,
    retr: RetrievalResult,
    *,
    llm: LLMClient | None = None,
    use_llm: bool = False,
) -> Answer:
    """生成最终答案。

    use_llm=True 且 llm 可用时，调用 LLM 在模板答案基础上重写更自然的中文回答；
    LLM 失败 / 未配置时静默退回模板答案。
    """
    ans = _template_compose(parsed, retr)
    if not use_llm or llm is None or not llm.available:
        return ans
    if not retr.triples and not retr.evidence_rows:
        return ans

    prompt = _build_llm_prompt(parsed, retr)
    enhanced = llm.complete(prompt)
    if enhanced.strip():
        ans.text = enhanced.strip()
        ans.llm_used = True
    else:
        if llm.last_error:
            ans.notes.append(f"LLM 调用失败已降级模板：{llm.last_error}")
    return ans
