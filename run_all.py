"""一键脚本：抽取 → 银标 → 评估（严格 + 宽松）→ 摘要。

工作流：
    1. 用 ans.py 导出 data/entities_by_type.json（确保词典与抽取一致）；
    2. 运行 run_extract.py（默认 mock LLM）抽取三元组；
    3. 运行 tools/export_silver_gold.py 生成 silver_triples.csv / silver_triples_loose.csv；
    4. 运行 evaluate_kg.py 做严格评估，再运行 loose 评估；
    5. 打印最终统计与文件清单。

用法：
    uv run python run_all.py                      # mock LLM 模式（默认）
    uv run python run_all.py --llm openai         # 启用 OpenAI 兼容 API（需环境变量）
    uv run python run_all.py --skip-extract       # 跳过抽取，只跑银标+评估
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
    p = argparse.ArgumentParser(description="一键运行：抽取→银标→评估")
    p.add_argument("--llm", default="mock", choices=["mock", "openai"], help="LLM 模式")
    p.add_argument("--llm-discover", action="store_true", help="启用按章发现补全（仅 openai）")
    p.add_argument("--skip-extract", action="store_true", help="跳过抽取（仅复跑银标+评估）")
    p.add_argument("--skip-silver", action="store_true", help="跳过银标生成（用现有 gold/）")
    p.add_argument("--skip-eval", action="store_true", help="跳过评估")
    p.add_argument("--quiet", action="store_true", help="抽取阶段静默")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable

    # 1) 同步词典 JSON（让抽取和银标使用同一份词表）
    if run([py, "tools/export_entities_by_type_json.py"], "Step 1/4: 导出领域词典 JSON") != 0:
        return 1

    # 2) 抽取
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
        if run(cmd, "Step 2/4: 抽取三元组") != 0:
            return 2

    # 3) 银标
    if not args.skip_silver:
        if run([py, "tools/export_silver_gold.py"], "Step 3/4: 生成银标 (触发词+类型+章节)") != 0:
            return 3

    # 4) 评估（严格 + 宽松双口径）
    if not args.skip_eval:
        if run(
            [py, "evaluate_kg.py",
             "--pred", "output/triples_with_meta.csv",
             "--gold", "gold/silver_triples.csv",
             "--report", "output/eval_report.txt"],
            "Step 4a/4: 严格评估（多关系类型）",
        ) != 0:
            return 4
        if run(
            [py, "evaluate_kg.py",
             "--pred", "output/triples_with_meta.csv",
             "--gold", "gold/silver_triples_loose.csv",
             "--loose",
             "--report", "output/eval_report_loose.txt"],
            "Step 4b/4: 宽松评估（仅头尾匹配）",
        ) != 0:
            return 4

    # 5) 汇总
    print()
    print("=" * 70)
    print("[run_all] 全部完成。输出清单：")
    print("=" * 70)
    expected = [
        "output/triples_with_meta.csv",
        "output/entities.csv",
        "output/extraction_stats.txt",
        "knowledge_graph.csv",
        "gold/silver_entities.txt",
        "gold/silver_triples.csv",
        "gold/silver_triples_loose.csv",
        "output/eval_report.txt",
        "output/eval_report_loose.txt",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
