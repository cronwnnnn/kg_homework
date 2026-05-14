"""一键脚本：抽取 → 评估 → 摘要。

工作流：
    1. 若 data/entities_by_type.json 不存在，则用 ans.py 导出（一次性）；
       已存在时**不会**覆盖（避免抹掉历史调优词典）；强制重生用 --regen-vocab。
    2. 运行 run_extract.py（默认 mock LLM）抽取三元组；
    3. 运行 evaluate_kg.py 用第四章人工金标 (gold/gold_triples_augmented.csv) 评估；
    4. 打印最终统计与文件清单。

用法：
    uv run python run_all.py                                  # mock LLM 模式（默认）
    uv run python run_all.py --llm openai --llm-discover      # 启用 OpenAI 兼容 API
    uv run python run_all.py --skip-extract                   # 跳过抽取，只跑评估
    uv run python run_all.py --regen-vocab                    # 强制从 ans.py 重新生成 JSON
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str], desc: str) -> int:
    """执行子进程，返回退出码。"""
    print()
    print("=" * 70)
    print(f"[run_all] {desc}")
    print(f"[run_all] $ {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.time()
    code = subprocess.call(cmd, cwd=ROOT)
    print(f"[run_all] -> exit={code}, elapsed={time.time() - t0:.1f}s")
    return code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="一键运行：抽取→评估")
    p.add_argument("--llm", default="mock", choices=["mock", "openai"], help="LLM 模式")
    p.add_argument("--llm-discover", action="store_true", help="启用按章发现补全（仅 openai）")
    p.add_argument("--skip-extract", action="store_true", help="跳过抽取（仅复跑评估）")
    p.add_argument("--skip-eval", action="store_true", help="跳过评估")
    p.add_argument("--quiet", action="store_true", help="抽取阶段静默")
    p.add_argument(
        "--regen-vocab",
        action="store_true",
        help="强制从 ans.py 重新生成 data/entities_by_type.json（默认仅在 JSON 缺失时生成，避免覆盖调优版本）",
    )
    p.add_argument(
        "--gold",
        default="gold/gold_triples_augmented.csv",
        help="评估用的人工金标 CSV（默认第四章 augmented 版）",
    )
    p.add_argument("--chapter", default="第4章", help="只评估某章（默认第4章）；置空则评估全部")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable

    vocab_json = os.path.join(ROOT, "data", "entities_by_type.json")
    if args.regen_vocab or not os.path.isfile(vocab_json):
        if run([py, "tools/export_entities_by_type_json.py"], "Step 1/3: 导出领域词典 JSON（首次或强制重生）") != 0:
            return 1
    else:
        print()
        print("=" * 70)
        print(f"[run_all] Step 1/3: 复用已有词典 {os.path.relpath(vocab_json, ROOT)}")
        print(f"[run_all] （想从 ans.py 重新生成请加 --regen-vocab）")
        print("=" * 70)

    if not args.skip_extract:
        cmd = [
            py, "run_extract.py",
            "--entities-json", "data/entities_by_type.json",
            "--llm", args.llm,
        ]
        if args.llm_discover and args.llm == "openai":
            cmd.append("--llm-discover")
        if args.quiet:
            cmd.append("--quiet")
        if run(cmd, "Step 2/3: 抽取三元组") != 0:
            return 2

    if not args.skip_eval:
        eval_cmd = [
            py, "evaluate_kg.py",
            "--pred", "output/triples_with_meta.csv",
            "--gold", args.gold,
            "--aliases-file", "data/aliases.json",
            "--exclude-relations", "discussed_in,co_occurs_with",
            "--report", "output/eval_report.txt",
        ]
        if args.chapter:
            eval_cmd.extend(["--chapter", args.chapter, "--include-global"])
        if run(eval_cmd, "Step 3/3: 第四章金标评估（4 口径）") != 0:
            return 3

    print()
    print("=" * 70)
    print("[run_all] 全部完成。输出清单：")
    print("=" * 70)
    expected = [
        "output/triples_with_meta.csv",
        "output/entities.csv",
        "output/extraction_stats.txt",
        "knowledge_graph.csv",
        "output/eval_report.txt",
        "output/eval_tp.csv",
        "output/eval_fp.csv",
        "output/eval_fn.csv",
    ]
    for f in expected:
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p) / 1024
            print(f"  ok  {f}  ({sz:.1f} KB)")
        else:
            print(f"  --  {f}  (缺失)")

    print()
    print("[run_all] 可视化：uv run streamlit run app_streamlit.py")
    print("[run_all] 问答助手：uv run streamlit run app_qa/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
