"""用 LLM 重新生成第四章的金标三元组（实验对比用）。

设计目标：
    用 LLM 严格按原文重新抽取一份金标，对比当前人工金标，看 F1 上限是否被人工金标的覆盖度限制。

约束：
    1. head/tail 必须是原文逐字子串（≤15 字）；
    2. relation 必须在 RelationOntology 内；
    3. 按段落分块调用，每段最多 12 条；
    4. 输出 CSV 与 gold_triples_augmented.csv 同格式 (head,relation,tail)。

运行示例：
    set OPENAI_API_KEY=sk-...
    set OPENAI_BASE_URL=https://api.deepseek.com/v1
    set OPENAI_MODEL=deepseek-chat
    uv run python tools/regenerate_gold_by_llm.py --chapter 第4章 --output gold/gold_triples_llm.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app_qa.llm_client import LLMClient  # noqa: E402
from extractors.preprocess import TextPreprocessor  # noqa: E402
from extractors.schema import RelationOntology  # noqa: E402


_PROMPT_TEMPLATE = """你是一位严谨的中文知识图谱标注专家。

任务：从下面这段中文飞行器工程论文中，抽取**所有事实性的三元组**（head, relation, tail）。

【段落原文（来自论文 {chapter}）】
{text}

【关系本体】（relation 必须从下面列表中精确选择一个英文标识符）：
{relations}

【严格要求】
1. **head 和 tail 必须是段落原文的逐字子串**，长度 2-15 字符；
2. **优先抽取领域核心实体之间的关系**（飞行器、机翼构型、参数、技术、人物、组织、材料、结构部件、性能指标、控制方法）；
3. **避免泛化或主观推断**——只抽取段落明确表达的事实关系；
4. **避免代词或弱概念**——head 不能是"它/这些/本节/研究内容/过程"等；
5. **每个三元组单独成行，CSV 格式**：head,relation,tail
6. 一段最多输出 15 条；如果段落没有可抽的关系则输出空。
7. 不要输出任何 head/tail 缺失、关系拼错、或带数字编号的额外内容。

输出（仅 CSV，无其他文字、无表头、无代码块标记）："""


def load_relations() -> list[str]:
    rels: set[str] = set()
    rels.update(RelationOntology.TRIGGER_TABLE.keys())
    rels.update(RelationOntology.INVERSE_RELATIONS.keys())
    rels.update(RelationOntology.INVERSE_RELATIONS.values())
    rels.update({
        "instance_of", "type_taxonomy", "increases", "equivalent_to",
        "approximately", "applies_to", "defined_as", "discussed_in",
    })
    return sorted(rels)


_LINE_RE = re.compile(r"^\s*([^,]+),\s*([^,]+),\s*(.+?)\s*$")


def parse_llm_output(text: str, paragraph: str, allowed_rels: set[str]) -> list[tuple[str, str, str]]:
    """从 LLM 输出解析 CSV 行，严格校验 head/tail 是 paragraph 子串。"""
    if not text:
        return []
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or line.lower().startswith("head"):
            continue
        line = line.lstrip("-*•◆ ").rstrip("。，；")
        m = _LINE_RE.match(line)
        if not m:
            continue
        head = m.group(1).strip().strip("\"'`()（）")
        rel = m.group(2).strip().strip("\"'`")
        tail = m.group(3).strip().strip("\"'`()（）")

        if not head or not rel or not tail:
            continue
        if rel not in allowed_rels:
            continue
        if len(head) < 2 or len(head) > 20 or len(tail) < 2 or len(tail) > 25:
            continue
        if head not in paragraph or tail not in paragraph:
            continue
        if head == tail:
            continue

        key = (head, rel, tail)
        if key in seen:
            continue
        seen.add(key)
        triples.append(key)
    return triples


def main() -> int:
    parser = argparse.ArgumentParser(description="用 LLM 重新生成金标三元组")
    parser.add_argument("--input", default="aftcln.txt", help="论文原文路径")
    parser.add_argument("--chapter", default="第4章", help="只处理指定章节关键字")
    parser.add_argument("--output", default="gold/gold_triples_llm.csv", help="输出 CSV 路径")
    parser.add_argument("--max-paragraphs", type=int, default=0, help="最多处理段落数（0=全部）")
    parser.add_argument("--max-per-paragraph", type=int, default=15, help="每段最多采纳条数")
    parser.add_argument("--sleep", type=float, default=0.3, help="段落间睡眠秒数")
    parser.add_argument("--cache", default="output/_llm_gold_cache.json", help="LLM 响应缓存路径")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[regen-gold] 找不到 {args.input}", file=sys.stderr)
        return 1

    client = LLMClient()
    if not client.available:
        print("[regen-gold] OPENAI_API_KEY 未配置，无法调用 LLM", file=sys.stderr)
        return 1

    raw = open(args.input, "r", encoding="utf-8").read()
    pp = TextPreprocessor()
    paragraphs = pp.split_paragraphs(raw)
    target = [p for p in paragraphs if args.chapter in (p.chapter or "")]
    if args.max_paragraphs > 0:
        target = target[: args.max_paragraphs]

    print(f"[regen-gold] {args.chapter} 段落数: {len(target)}")
    if not target:
        print("[regen-gold] 没有匹配章节的段落", file=sys.stderr)
        return 1

    allowed_rels = set(load_relations())
    relation_block = ", ".join(sorted(allowed_rels))

    cache: dict[str, str] = {}
    if os.path.exists(args.cache):
        try:
            with open(args.cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    all_triples: list[tuple[str, str, str, str]] = []

    for i, para in enumerate(target):
        cache_key = f"{para.chapter}|{para.paragraph_id}|{hash(para.text) & 0xffffffff}"
        if cache_key in cache:
            print(f"[regen-gold] [{i+1}/{len(target)}] 命中缓存 para#{para.paragraph_id} len={len(para.text)}")
            output = cache[cache_key]
        else:
            prompt = _PROMPT_TEMPLATE.format(
                chapter=para.chapter,
                text=para.text,
                relations=relation_block,
            )
            output = client.complete(prompt, temperature=0.1, max_tokens=1500)
            if not output and client.last_error:
                print(f"[regen-gold] LLM 报错: {client.last_error}", file=sys.stderr)
            cache[cache_key] = output or ""
            with open(args.cache, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            time.sleep(args.sleep)
            print(f"[regen-gold] [{i+1}/{len(target)}] LLM 响应 para#{para.paragraph_id} len={len(para.text)} -> 字数 {len(output)}")

        parsed = parse_llm_output(output, para.text, allowed_rels)
        parsed = parsed[: args.max_per_paragraph]
        for h, r, t in parsed:
            all_triples.append((h, r, t, para.chapter))
        if parsed:
            print(f"           采纳 {len(parsed)} 条三元组")

    seen_triple: set[tuple[str, str, str]] = set()
    dedup: list[tuple[str, str, str, str]] = []
    for h, r, t, ch in all_triples:
        k = (h, r, t)
        if k in seen_triple:
            continue
        seen_triple.add(k)
        dedup.append((h, r, t, ch))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "relation", "tail"])
        for h, r, t, _ in dedup:
            writer.writerow([h, r, t])

    print()
    print("=" * 60)
    print(f"AI 金标输出: {args.output}")
    print(f"总三元组数: {len(dedup)} 条")
    from collections import Counter
    rel_counter = Counter(r for _, r, _, _ in dedup)
    print(f"关系分布 Top-15:")
    for r, c in rel_counter.most_common(15):
        print(f"  {r:25s} {c}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
