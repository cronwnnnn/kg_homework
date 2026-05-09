"""LLM 增强层：在传统抽取候选基础上做关系归一化、过滤、隐式关系补全。

关键约束（来自任务要求）：
    - 不允许只用 LLM 抽取——LLM 必须基于"传统层已识别的实体"工作；
    - 必须可降级为 mock 模式，便于本地无 API 时也能跑通流水线；
    - 真实模式兼容 OpenAI 协议（DeepSeek、Kimi、智谱、Ollama OpenAI-Compatible 都行）。
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable, Sequence

from .relation_normalizer import RelationNormalizer
from .schema import Triple


_LLM_PROMPT = """你是一个知识图谱三元组质检员。
我已经用传统 NLP 算法从一段中文论文中提取出若干候选三元组。
请你做三件事：
1) 删除明显错误或语义重复的三元组（保留更合理的那条即可）；
2) 把每条三元组的 relation 字段统一为我提供的本体类型；
3) 仅当候选实体集合中已经存在的实体之间存在明显的额外关系时，可补充少量三元组（每段最多补 2 条）。

【关系本体】(relation_type → 中文含义)
- has_part / part_of / is_a / instance_of / used_for / applied_to
- improves / reduces / causes / leads_to / enables / prevents
- depends_on / affects / controls / driven_by / actuated_by
- manufactures / develops / developed_by / originates_from
- greater_than / less_than / equals_to / has_value / approximately
- located_at / connected_to / combines_with / transforms_to / generates
- uses_method / verifies / implements / satisfies / needs / solves / similar_to / related_to

【输入】
段落文本：
{text}

候选实体集合：
{entities}

候选三元组（CSV，每行 head,relation,tail,source,score）：
{candidates}

【输出格式】严格 JSON，无任何额外说明文字：
{{
  "kept": [{{"head": "...", "relation": "...", "tail": "...", "score": 0-1}} , ...],
  "added": [{{"head": "...", "relation": "...", "tail": "...", "score": 0-1}} , ...]
}}
"""


def _allowed_relation_labels() -> frozenset[str]:
    """LLM 发现阶段允许写入的关系名（与流水线本体一致）。"""
    from .schema import RelationOntology

    names: set[str] = set(RelationOntology.TRIGGER_TABLE)
    names.update(RelationOntology.INVERSE_RELATIONS)
    names.update(RelationOntology.INVERSE_RELATIONS.values())
    names.update(
        {
            "instance_of",
            "discussed_in",
            "related_to",
            "has_parameter",
            "has_performance",
            "has_configuration",
            "uses_material",
            "affiliated_with",
            "applied_to",
            "co_occurs_with",
            "part_of",
            "controlled_by",
            "enables",
            "generated_by",
            "transformed_from",
            "approximately",
            "developed_by",
        }
    )
    return frozenset(names)


_LLM_DISCOVER_PROMPT = """你是知识图谱「发现补全」助手，任务是阅读下面一整章/一节的论文中文正文，
在**不重复已有候选三元组**的前提下，找出正文中**明确可支持**的新知识三元组。

硬性约束（违反则该条无效）：
1) head 与 tail 必须是下面【正文】中**连续出现的子串**（逐字可查），不得编造正文中不存在的实体名称。
2) relation 必须是下列英文键之一：{allowed_relations}
3) 每条三元组必须在正文中有清晰语义依据；宁可少报，不要猜测。
4) 最多输出 {max_new} 条**新增**三元组（与已有候选 head,relation,tail 完全相同的不要输出）。

【正文】
{text}

【已有候选三元组】（head,relation,tail；勿重复）
{existing}

请严格输出 JSON（不要 markdown 围栏），格式如下：
{{"new_triples":[{{"head":"...","relation":"...","tail":"...","score":0.85}}, ...]}}
若无可补充则输出 {{"new_triples":[]}}。
"""


class LLMEnhancer:
    """LLM 协同模块。

    模式：
        - "mock"：不调用任何 API，等价于"原样透传 + 关系再归一化 + 简单去重"；
        - "openai"：通过 OpenAI 兼容协议调用真实 LLM。
    """

    def __init__(
        self,
        mode: str = "mock",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_batch: int = 50,
    ) -> None:
        self.mode = mode
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        mb_env = os.environ.get("KG_LLM_MAX_BATCH")
        if mb_env:
            try:
                max_batch = max(5, int(mb_env))
            except ValueError:
                pass
        self.max_batch = max_batch
        self.normalizer = RelationNormalizer()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs) if kwargs else OpenAI()
            return self._client
        except Exception as exc:  # noqa: BLE001
            print(f"[llm_enhancer] OpenAI 客户端初始化失败：{exc}，自动回退 mock。")
            self.mode = "mock"
            return None

    def enhance(self, candidates: Sequence[Triple], paragraph_text: str = "", entities: Sequence[str] = ()) -> list[Triple]:
        """对一批候选三元组做增强。"""
        if not candidates:
            return []
        if self.mode == "mock":
            return self._mock_enhance(candidates)
        # openai：按 max_batch 分块送审，避免每章只处理前 30 条、其余被静默丢弃
        cands = list(candidates)
        out: list[Triple] = []
        for i in range(0, len(cands), self.max_batch):
            chunk = cands[i : i + self.max_batch]
            out.extend(self._llm_enhance(chunk, paragraph_text, entities))
        return out

    def enhance_grouped(
        self,
        groups: Iterable[tuple[str, Sequence[str], Sequence[Triple]]],
    ) -> list[Triple]:
        """按段落分组增强：每段一次 LLM 调用。"""
        out: list[Triple] = []
        for text, ents, cands in groups:
            out.extend(self.enhance(cands, paragraph_text=text, entities=ents))
        return out

    def discover_novel_triples(
        self,
        chapter_text: str,
        chapter: str,
        existing_triples: Sequence[Triple],
        max_new: int = 50,
        max_existing_lines: int = 60,
    ) -> list[Triple]:
        """在整章正文上请求 LLM 补充新三元组；仅 openai 模式；结果经「子串在正文」校验。"""
        if self.mode != "openai":
            return []
        client = self._ensure_client()
        if client is None:
            return []

        allowed = sorted(_allowed_relation_labels())
        rel_sample = ", ".join(allowed[:45]) + ("..." if len(allowed) > 45 else "")
        lines = [f"{t.head},{t.relation},{t.tail}" for t in existing_triples[:max_existing_lines]]
        limit = LLMEnhancer._discovery_text_limit()
        text_snip = chapter_text[:limit]
        prompt = _LLM_DISCOVER_PROMPT.format(
            allowed_relations=rel_sample,
            max_new=max_new,
            text=text_snip,
            existing="\n".join(lines) if lines else "(无)",
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
            )
            content = resp.choices[0].message.content or ""
            data = self._parse_json(content)
        except Exception as exc:  # noqa: BLE001
            print(f"[llm_enhancer] discover 调用失败：{exc}")
            return []

        raw_items = data.get("new_triples", [])
        if not isinstance(raw_items, list):
            return []

        allowed_set = _allowed_relation_labels()
        out: list[Triple] = []
        seen: set[tuple[str, str, str]] = {(t.head, t.relation, t.tail) for t in existing_triples}
        body = text_snip
        for item in raw_items:
            if len(out) >= max_new:
                break
            try:
                h = str(item.get("head", "")).strip()
                r = str(item.get("relation", "")).strip()
                t = str(item.get("tail", "")).strip()
                s = float(item.get("score", 0.65))
            except Exception:  # noqa: BLE001
                continue
            if len(h) < 2 or len(t) < 2 or h == t:
                continue
            if r not in allowed_set:
                continue
            if h not in body or t not in body:
                continue
            key = (h, r, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Triple(
                    head=h,
                    relation=r,
                    tail=t,
                    trigger="llm_discover",
                    source="llm_discover",
                    score=max(0.0, min(1.0, s)),
                    chapter=chapter,
                    sentence=body[:220],
                )
            )
        return out

    @staticmethod
    def _discovery_text_limit() -> int:
        return int(os.environ.get("KG_LLM_DISCOVER_CHARS", "16000"))

    # ---------- mock ----------

    def _mock_enhance(self, candidates: Sequence[Triple]) -> list[Triple]:
        """Mock 模式：保持各抽取器已经设定好的关系类型，仅做去重 + 边界清洗。

        注：不再对 relation 做二次归一化——上游每个抽取器已经映射到本体。
        """
        seen: set[tuple[str, str, str]] = set()
        out: list[Triple] = []
        for tri in candidates:
            head = tri.head.strip()
            tail = tri.tail.strip()
            if not head or not tail or head == tail:
                continue
            key = (head, tri.relation, tail)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Triple(
                    head=head,
                    relation=tri.relation,
                    tail=tail,
                    trigger=tri.trigger,
                    source=tri.source,
                    score=tri.score,
                    chapter=tri.chapter,
                    sentence=tri.sentence,
                )
            )
        return out

    # ---------- 真实 LLM ----------

    def _llm_enhance(self, candidates: Sequence[Triple], paragraph_text: str, entities: Sequence[str]) -> list[Triple]:
        client = self._ensure_client()
        if client is None:
            return self._mock_enhance(candidates)

        cand_lines = [f"{t.head},{t.relation},{t.tail},{t.source},{t.score}" for t in candidates[: self.max_batch]]
        polish_chars = int(os.environ.get("KG_LLM_POLISH_CHARS", "16000"))
        prompt = _LLM_PROMPT.format(
            text=paragraph_text[:polish_chars],
            entities=", ".join(sorted(set(entities))[:80]),
            candidates="\n".join(cand_lines),
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content or ""
            data = self._parse_json(content)
        except Exception as exc:  # noqa: BLE001
            print(f"[llm_enhancer] 调用失败：{exc}，本批回退 mock。")
            return self._mock_enhance(candidates)

        chapter = candidates[0].chapter if candidates else ""
        sentence = candidates[0].sentence if candidates else ""
        kept_raw = data.get("kept", [])
        added_raw = data.get("added", [])
        kept_list = kept_raw if isinstance(kept_raw, list) else []
        added_list = added_raw if isinstance(added_raw, list) else []
        out: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for is_added, item in [(False, x) for x in kept_list] + [(True, x) for x in added_list]:
            try:
                h = str(item.get("head", "")).strip()
                r = str(item.get("relation", "")).strip()
                t = str(item.get("tail", "")).strip()
                s = float(item.get("score", 0.7))
            except Exception:  # noqa: BLE001
                continue
            if not h or not r or not t or h == t:
                continue
            key = (h, r, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Triple(
                    head=h,
                    relation=r,
                    tail=t,
                    trigger="llm",
                    source="llm" if is_added else "trigger+llm",
                    score=max(0.0, min(1.0, s)),
                    chapter=chapter,
                    sentence=sentence,
                )
            )
        if not out:
            # JSON 解析失败或模型未返回有效项：保留本批原始候选，避免整章被清空
            return self._mock_enhance(candidates)
        return out

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        try:
            return json.loads(content)
        except Exception:  # noqa: BLE001
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:  # noqa: BLE001
                    return {}
            return {}
