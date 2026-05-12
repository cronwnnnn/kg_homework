"""LLM NER 兜底：让大模型扫论文找词典未登录的专业术语。

使用方法（PowerShell）::

    $env:OPENAI_API_KEY = "sk-xxx"
    $env:OPENAI_BASE_URL = "https://api.deepseek.com"
    $env:OPENAI_MODEL = "deepseek-chat"
    uv run python tools/llm_ner_expand.py --dry-run        # 先跑前 3 段试水
    uv run python tools/llm_ner_expand.py                  # 跑全文

输出：
    data/entities_to_add_llm.json      候选清单（带 evidence/reason，待审查）
    data/llm_ner_expand_log.txt        每段调用日志（便于排错）

后续：assistant 会读 entities_to_add_llm.json 做人工 review，
      过滤明显错误，输出 entities_to_add_llm_audited.json，
      再用 tools/merge_ner_terms.py 合并进 entities_by_type.json。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOWED_TYPES: list[str] = [
    "AIRCRAFT",
    "WING_CONFIGURATION",
    "PARAMETER",
    "AERODYNAMIC_CONCEPT",
    "STRUCTURAL_COMPONENT",
    "CONTROL_METHOD",
    "PERFORMANCE_METRIC",
    "ORGANIZATION",
    "PERSON",
    "TECHNOLOGY",
    "MATERIAL",
    "FLIGHT_PHASE",
    "EQUATION",
    "CONCEPT",
]

TYPE_HINTS: dict[str, str] = {
    "AIRCRAFT": "具体飞行器型号或代号（含字母数字编号、绰号），如 RQ-4、F-14、波音 737、太阳神",
    "WING_CONFIGURATION": "机翼布局/构型/翼面形式，如 大展弦比平直机翼、连接翼布局、折叠翼、变后掠翼",
    "PARAMETER": "可测量的物理量/气动参数，名词性，如 展弦比、升阻比、翼展、迎角、雷诺数（不要带数值）",
    "AERODYNAMIC_CONCEPT": "气动现象/概念，如 诱导阻力、激波、气动弹性、湍流、附面层",
    "STRUCTURAL_COMPONENT": "飞机结构部件，如 机翼、副翼、尾翼、襟翼、桁架、蒙皮",
    "CONTROL_METHOD": "控制律/控制方法，如 PID、LQR、模型预测控制、滑模控制",
    "PERFORMANCE_METRIC": "性能/任务指标，如 航程、续航时间、最大起飞重量、有效载荷",
    "ORGANIZATION": "研究机构/公司/军方单位，如 NASA、洛克希德马丁、波音、中国航空工业集团",
    "PERSON": "人名（中外作者、科学家、设计师）",
    "TECHNOLOGY": "技术方案/方法学/工艺，如 滑动蒙皮、电作动器、伸缩翼、桁架支撑",
    "MATERIAL": "材料，如 复合材料、碳纤维、铝合金、钛合金",
    "FLIGHT_PHASE": "飞行阶段，如 巡航、起飞、着陆、超声速飞行、亚声速飞行",
    "EQUATION": "明确命名的公式/方程",
    "CONCEPT": "其他领域概念/方法论性术语（兜底类型，慎用）",
}


_CHAPTER_HEAD = re.compile(r"^第\s*[一二三四五六七八九十百0-9]+\s*章")
_SECTION_HEAD = re.compile(r"^\d+(?:\.\d+){0,3}\s")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, target_chars: int = 1800) -> list[tuple[str, str]]:
    """把论文切成 (chapter_label, chunk_text) 列表。

    切分规则：
        1) 先按章节标题（"第N章" 或 "N.M..." 数字开头）划块；
        2) 同一节内若过长，按段落继续切到 ~target_chars。
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_label = "导言"
    current_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _CHAPTER_HEAD.match(stripped) or _SECTION_HEAD.match(stripped):
            if current_lines:
                sections.append((current_label, current_lines))
            current_label = stripped[:60]
            current_lines = []
        else:
            if stripped:
                current_lines.append(line)
    if current_lines:
        sections.append((current_label, current_lines))

    chunks: list[tuple[str, str]] = []
    for label, body in sections:
        buf: list[str] = []
        buf_len = 0
        idx = 0
        for line in body:
            if buf and buf_len + len(line) > target_chars:
                chunks.append((f"{label} #{idx + 1}", "\n".join(buf)))
                idx += 1
                buf = []
                buf_len = 0
            buf.append(line)
            buf_len += len(line)
        if buf:
            suffix = f" #{idx + 1}" if idx else ""
            chunks.append((f"{label}{suffix}", "\n".join(buf)))
    return chunks


def load_known_terms(entities_json: str) -> tuple[set[str], dict[str, list[str]]]:
    with open(entities_json, encoding="utf-8-sig") as f:
        data: dict[str, list[str]] = json.load(f)
    known: set[str] = set()
    for vs in data.values():
        for w in vs:
            w = (w or "").strip()
            if w:
                known.add(w)
    return known, data


def _build_prompt(chunk_text: str, samples_by_type: dict[str, list[str]], max_new: int) -> str:
    type_lines = []
    for t in ALLOWED_TYPES:
        hint = TYPE_HINTS.get(t, "")
        examples = "、".join(samples_by_type.get(t, [])[:6])
        type_lines.append(f"- {t}（{hint}）；现有示例：{examples}")
    type_block = "\n".join(type_lines)

    return f"""你是航空领域知识图谱标注专家。下面是一段中文论文正文。

任务：找出**正文中出现、但不属于以下示例的专业术语**，并按类型分类。

【可用类型与示例】
{type_block}

【硬性规则】
1. 输出的 term 必须是正文中**逐字出现**的连续子串（不要做翻译/改写）；
2. 不输出纯数字/年份/单位（如 "12 m"、"2020 年"、"30%"），数值是属性不是实体；
3. 不输出已经在示例里出现过的术语（避免重复）；
4. 不输出单字、过短或过宽的词（如 "飞机"、"性能"、"研究"），要选**专业、独立、可作为知识图谱节点**的术语；
5. 每个 term 必须给出 evidence（正文 ≤80 字片段，含该 term）和 reason（1 句话说明为什么是这类）；
6. 优先选高质量术语，宁少勿多，最多 {max_new} 条；
7. type 必须是上面列表中的英文名之一，不要新造类型。

【正文】
{chunk_text}

【输出格式】严格 JSON（无 markdown 围栏、无解释文字）：
{{"new_terms":[
  {{"term":"...","type":"AIRCRAFT","evidence":"...","reason":"..."}},
  ...
]}}
若无新术语则输出 {{"new_terms":[]}}。
"""


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


_VALUE_TAIL = re.compile(r"\d+(?:\.\d+)?\s*(?:m|km|kg|t|°|度|%|倍|m/s)?$")


def _is_garbage(term: str) -> bool:
    t = (term or "").strip()
    if len(t) < 2:
        return True
    if t.isdigit():
        return True
    if _VALUE_TAIL.fullmatch(t):
        return True
    if t in {"飞机", "飞行器", "性能", "研究", "技术", "方法", "结构", "影响", "问题"}:
        return True
    return False


def _samples_by_type(existing: dict[str, list[str]]) -> dict[str, list[str]]:
    """每个类型挑前 30 个作为 prompt 里的示例（不全列出，避免 prompt 太长）。"""
    out: dict[str, list[str]] = {}
    for t in ALLOWED_TYPES:
        vs = existing.get(t, [])
        out[t] = vs[:30]
    return out


def run(args: argparse.Namespace) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    text = _read_text(args.text)
    chunks = split_into_chunks(text, target_chars=args.chunk_chars)
    if args.dry_run:
        chunks = chunks[: args.dry_limit]
    elif args.limit > 0:
        chunks = chunks[: args.limit]

    print(f"[llm_ner] 切片完成：{len(chunks)} 段（目标长度 {args.chunk_chars} 字/段）")

    known, existing = load_known_terms(args.dict)
    samples = _samples_by_type(existing)
    print(f"[llm_ner] 已知实体 {len(known)} 个，14 个类型")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        print("[llm_ner] 未检测到 OPENAI_API_KEY，无法调用 LLM；请先设置环境变量。")
        return 1
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or ""
    model = args.model or os.environ.get("OPENAI_MODEL") or "deepseek-chat"
    print(f"[llm_ner] 调用：model={model}  base_url={base_url or '(默认)'}")

    from openai import OpenAI  # type: ignore

    kwargs: dict = {"timeout": args.timeout}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    log_path = os.path.join(ROOT, "data", "llm_ner_expand_log.txt")
    out_path = args.out or os.path.join(ROOT, "data", "entities_to_add_llm.json")
    log_f = open(log_path, "w", encoding="utf-8")

    aggregated: dict[str, list[dict]] = {t: [] for t in ALLOWED_TYPES}
    seen_terms: set[str] = set()
    ok_chunks = 0
    err_chunks = 0
    total_new = 0

    for idx, (label, chunk_text) in enumerate(chunks, 1):
        prompt = _build_prompt(chunk_text, samples, max_new=args.max_per_chunk)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            err_chunks += 1
            msg = f"[chunk {idx}/{len(chunks)}] {label}  ERROR: {exc}"
            print(msg)
            log_f.write(msg + "\n")
            time.sleep(args.sleep_on_error)
            continue

        data = _parse_json(content)
        items = data.get("new_terms", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            items = []

        new_in_chunk = 0
        for item in items:
            try:
                term = str(item.get("term", "")).strip()
                etype = str(item.get("type", "")).strip().upper()
                evidence = str(item.get("evidence", "")).strip()[:160]
                reason = str(item.get("reason", "")).strip()[:120]
            except Exception:  # noqa: BLE001
                continue
            if etype not in ALLOWED_TYPES:
                continue
            if _is_garbage(term):
                continue
            if term in known or term in seen_terms:
                continue
            if term not in chunk_text:
                continue
            aggregated[etype].append(
                {
                    "term": term,
                    "evidence": evidence,
                    "reason": reason,
                    "source": "llm",
                    "chapter": label,
                }
            )
            seen_terms.add(term)
            new_in_chunk += 1
            total_new += 1

        ok_chunks += 1
        msg = f"[chunk {idx}/{len(chunks)}] {label}  -> +{new_in_chunk} 新术语（累计 {total_new}）"
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

        if args.sleep > 0 and idx < len(chunks):
            time.sleep(args.sleep)

    summary = {
        "total_chunks": len(chunks),
        "ok_chunks": ok_chunks,
        "err_chunks": err_chunks,
        "total_new_terms": total_new,
        "by_type": {t: len(v) for t, v in aggregated.items() if v},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    log_f.write("\n=== SUMMARY ===\n")
    log_f.write(json.dumps(summary, ensure_ascii=False, indent=2))
    log_f.close()

    print("\n=== 完成 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n候选写入: {out_path}")
    print(f"日志写入: {log_path}")
    print("下一步：让 assistant 读这份 JSON，逐项 review，过滤明显错的，再合并。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM NER 兜底：扫论文找词典外术语")
    p.add_argument("--text", default=os.path.join(ROOT, "aftcln.txt"), help="论文文本路径")
    p.add_argument("--dict", default=os.path.join(ROOT, "data", "entities_by_type.json"), help="现有实体词典")
    p.add_argument("--out", default="", help="输出 JSON 路径（默认 data/entities_to_add_llm.json）")
    p.add_argument("--chunk-chars", type=int, default=1800, help="单段目标长度（字符）")
    p.add_argument("--max-per-chunk", type=int, default=12, help="单段最多输出新术语数")
    p.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 段（不与 --dry-run 同用时优先）")
    p.add_argument("--dry-run", action="store_true", help="只跑前几段（默认 3 段）用于试水")
    p.add_argument("--dry-limit", type=int, default=3, help="--dry-run 模式下的段数")
    p.add_argument("--sleep", type=float, default=0.4, help="相邻调用之间的间隔（秒）")
    p.add_argument("--sleep-on-error", type=float, default=2.0, help="单段失败后的间隔（秒）")
    p.add_argument("--timeout", type=float, default=120.0, help="LLM 调用超时（秒）")
    p.add_argument("--api-key", default="", help="覆盖 OPENAI_API_KEY")
    p.add_argument("--base-url", default="", help="覆盖 OPENAI_BASE_URL")
    p.add_argument("--model", default="", help="覆盖 OPENAI_MODEL")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
