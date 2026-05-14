"""自动抽取主入口：从论文 aftcln.txt 抽取知识图谱三元组。

运行示例：
    # 默认 mock 模式（无须任何 API），词典来自 ans.EntityLibrary
    uv run python run_extract.py

    # 不依赖运行时 ans：先用工具导出 JSON，再指定路径（推荐用于交付/解耦）
    uv run python tools/export_entities_by_type_json.py
    uv run python run_extract.py --entities-json data/entities_by_type.json

    # 接入 OpenAI 兼容 API（DeepSeek / Kimi / 智谱 等）
    set OPENAI_API_KEY=sk-xxx
    set OPENAI_BASE_URL=https://api.deepseek.com/v1
    set OPENAI_MODEL=deepseek-chat
    uv run python run_extract.py --llm openai --entities-json data/entities_by_type.json
    # 二阶段：按章发现正文中可核对的新三元组（须 openai）
    uv run python run_extract.py --llm openai --llm-discover --entities-json data/entities_by_type.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from extractors import ExtractionPipeline
from extractors.pipeline import PipelineConfig
from extractors.preprocess import TextPreprocessor


_EXCLUDED_VOCAB_ETYPES = {"NUMERIC_VALUE"}
_EXCLUDED_VOCAB_TERMS = {
    # 过短/过宽，会导致 NER 在每句话都误命中
    "是", "为", "在", "有", "无", "其", "之", "者",
    # 这些词义太弱，作为实体没有信息量
    "部分", "情况", "状态", "方式", "方法", "需要", "需求", "要求",
    "影响", "决定", "可能", "进而", "因此", "通过", "使用", "采用", "利用",
    "来自", "之间", "两者", "三者", "其他", "其它", "上述", "下述",
    "目前", "近年", "随着", "随之", "结果", "结论", "讨论", "分析",
    "研究", "本文", "本章", "本节", "公式", "图表", "表格", "样品",
    "单一", "复杂",
}


def load_domain_vocab() -> tuple[list[str], dict[str, list[str]]]:
    """从 ans.py 加载领域实体词典。

    返回：
        (flat_vocab, entities_by_type)
        - flat_vocab：扁平词表（用于 NER 匹配）。
        - entities_by_type：按类型分组的实体（用于 TypeBasedExtractor / CooccurrenceTypeExtractor）。

    过滤规则：
        - 排除 NUMERIC_VALUE 类型（避免 "0.4"、"10%" 被当成实体匹配，造成 is_a 大量噪声）；
        - 排除"是/为/在"等过短虚词；
        - 排除"部分/情况/方式"等无信息量的高频词。
    """
    try:
        from ans import EntityLibrary  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[run_extract] 加载 ans.EntityLibrary 失败：{exc}")
        return [], {}
    entities_dict = EntityLibrary.get_all_entities()
    flat: set[str] = set()
    entities_by_type: dict[str, list[str]] = {}
    for etype, elist in entities_dict.items():
        type_name = getattr(etype, "name", "")
        if type_name in _EXCLUDED_VOCAB_ETYPES:
            continue
        cleaned: list[str] = []
        for w in elist:
            ww = (w or "").strip()
            if not ww or len(ww) < 2 or ww in _EXCLUDED_VOCAB_TERMS:
                continue
            if ww.replace(".", "").replace("%", "").isdigit():
                continue
            flat.add(ww)
            cleaned.append(ww)
        if cleaned:
            entities_by_type[type_name] = cleaned
    return sorted(flat), entities_by_type


def load_entities_from_json(path: str) -> tuple[list[str], dict[str, list[str]]]:
    """从 JSON 加载按类型分组的实体（运行时不再 import ans）。

    文件格式：与 `tools/export_entities_by_type_json.py` 导出一致，
    顶层为对象，键为类型名（如 AIRCRAFT），值为字符串数组。
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        print(f"[run_extract] JSON 根节点必须是对象：{path}", file=sys.stderr)
        return [], {}
    flat: set[str] = set()
    entities_by_type: dict[str, list[str]] = {}
    for type_name, elist in raw.items():
        if type_name in _EXCLUDED_VOCAB_ETYPES:
            continue
        if not isinstance(elist, list):
            continue
        cleaned: list[str] = []
        for w in elist:
            ww = (w or "").strip()
            if not ww or len(ww) < 2 or ww in _EXCLUDED_VOCAB_TERMS:
                continue
            if ww.replace(".", "").replace("%", "").isdigit():
                continue
            flat.add(ww)
            cleaned.append(ww)
        if cleaned:
            entities_by_type[str(type_name)] = cleaned
    return sorted(flat), entities_by_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于论文的中文知识图谱三元组抽取流水线")
    parser.add_argument(
        "--entities-json",
        default=os.environ.get("KG_ENTITIES_JSON"),
        metavar="PATH",
        help="从该 JSON 加载领域实体（可脱离 ans.py）；未指定时仍从 ans.EntityLibrary 加载",
    )
    parser.add_argument("--input", default="aftcln.txt", help="输入论文文本路径")
    parser.add_argument("--llm", default=os.environ.get("KG_LLM_MODE", "mock"), choices=["mock", "openai"], help="LLM 模式")
    parser.add_argument("--enable-svo", action="store_true", help="启用 spaCy SVO 抽取（默认关闭：未装模型时无产出，装上后噪声较大）")
    parser.add_argument("--no-dep-re", action="store_true", help="禁用依存增强抽取 (DependencyREExtractor)，默认开启")
    parser.add_argument("--no-llm", action="store_true", help="完全跳过 LLM 增强层")
    parser.add_argument("--max-window", type=int, default=30, help="触发词共现窗口大小")
    parser.add_argument("--min-score", type=float, default=0.30, help="触发词三元组最小置信度")
    parser.add_argument("--output-csv", default="knowledge_graph.csv", help="兼容老格式的简版 CSV 输出路径")
    parser.add_argument(
        "--llm-discover",
        action="store_true",
        help="在 --llm openai 时启用二阶段：按章把正文+候选三元组发给模型，补充可在原文子串核对的新实体关系",
    )
    parser.add_argument(
        "--llm-discover-max-new",
        type=int,
        default=40,
        metavar="N",
        help="每章 LLM 发现补全最多采纳的新增三元组条数（默认 40，便于冲关系≥1000、实体≥500；可调低降噪）",
    )
    parser.add_argument(
        "--llm-discover-max-lines",
        type=int,
        default=60,
        metavar="N",
        help="传给模型的已有候选三元组最多行数（默认 60）",
    )
    parser.add_argument(
        "--paper-entity-min-freq",
        type=int,
        default=1,
        metavar="N",
        help="论文挖掘出的实体串在全文至少出现 N 次才并入词典（默认 1；设为 2 可降噪）",
    )
    parser.add_argument(
        "--enable-paper-entity-mine",
        action="store_true",
        help="启用基于 spaCy 的论文实体挖掘（默认关闭：会引入长复合实体吞并短实体的 FP）",
    )
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[run_extract] 找不到输入文件：{args.input}", file=sys.stderr)
        return 1

    print(f"[run_extract] 加载领域实体词典 ...")
    if args.entities_json:
        if not os.path.isfile(args.entities_json):
            print(f"[run_extract] 找不到 --entities-json 文件：{args.entities_json}", file=sys.stderr)
            return 1
        vocab, entities_by_type = load_entities_from_json(args.entities_json)
        print(f"[run_extract] 已从 JSON 加载：{args.entities_json}")
    else:
        vocab, entities_by_type = load_domain_vocab()
    print(f"[run_extract] 领域实体数: {len(vocab)} (分 {len(entities_by_type)} 类)")
    if len(vocab) < 100:
        print("[run_extract] 警告：领域词典实体过少，触发词共现抽取的召回会较低。")

    if args.llm_discover and args.llm != "openai":
        print(
            "[run_extract] 提示：--llm-discover 仅在 --llm openai 且能访问 API 时生效；"
            "当前为 mock 时将自动跳过发现补全。",
            file=sys.stderr,
        )

    print(f"[run_extract] 读取论文：{args.input}")
    raw_text = TextPreprocessor.load_text(args.input)
    print(f"[run_extract] 论文字符数: {len(raw_text)}")

    config = PipelineConfig(
        use_svo=args.enable_svo,
        use_dep_re=not args.no_dep_re,
        use_llm=not args.no_llm,
        llm_mode=args.llm,
        trigger_max_window=args.max_window,
        trigger_min_score=args.min_score,
        csv_simple_path=args.output_csv,
        use_llm_discovery=args.llm_discover,
        llm_discovery_max_new_per_chapter=args.llm_discover_max_new,
        llm_discovery_max_existing_lines=args.llm_discover_max_lines,
        mine_paper_entities=args.enable_paper_entity_mine,
        paper_entity_min_doc_freq=max(1, args.paper_entity_min_freq),
    )
    pipeline = ExtractionPipeline(vocab=vocab, config=config, entities_by_type=entities_by_type)

    result = pipeline.run(raw_text, verbose=not args.quiet)
    pipeline.export(result)

    print()
    print("=" * 60)
    print(f"实体数: {result.stats['entity_count']}")
    print(f"三元组数: {result.stats['triple_count']}")
    print(f"按来源: {result.stats['by_source']}")
    if result.stats.get("paper_mined_candidates", 0) or result.stats.get("paper_mined_added_to_vocab", 0):
        print(
            f"论文实体挖掘: 候选 {result.stats.get('paper_mined_candidates', 0)} 串, "
            f"新词入库 {result.stats.get('paper_mined_added_to_vocab', 0)}"
        )
    print(f"Top-10 关系类型: {sorted(result.stats['by_relation'].items(), key=lambda x: -x[1])[:10]}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
