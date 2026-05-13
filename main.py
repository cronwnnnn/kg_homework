"""项目主入口：子命令转发到各脚本。

子命令：
    all     一键执行：抽取 → 评估
    extract 仅从 aftcln.txt 自动抽取（run_extract.py）
    manual  打印 ans.py 领域词典统计
    eval    与人工金标比对算 P/R/F1（evaluate_kg.py）
    app     启动 Streamlit 图谱浏览（app_streamlit.py）

示例：
    uv run python main.py all                  # 一键全跑
    uv run python main.py extract --quiet
    uv run python main.py manual
    uv run python main.py eval --chapter "第4章" --include-global
    uv run python main.py app
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单-双折叠翼变体飞行器知识图谱构建系统（抽取/评估/可视化）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("all", help="一键执行：抽取→评估")
    p_all.add_argument("rest", nargs=argparse.REMAINDER, help="透传给 run_all.py")

    p_ext = sub.add_parser("extract", help="抽取（run_extract.py）")
    p_ext.add_argument("rest", nargs=argparse.REMAINDER)

    sub.add_parser("manual", help="打印 ans.py 词典统计")

    p_eval = sub.add_parser("eval", help="评估（evaluate_kg.py）")
    p_eval.add_argument("rest", nargs=argparse.REMAINDER)

    p_app = sub.add_parser("app", help="Streamlit 可视化")
    p_app.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    py = sys.executable

    if args.cmd == "all":
        return subprocess.call([py, "run_all.py", *args.rest])
    if args.cmd == "extract":
        return subprocess.call([py, "run_extract.py", *args.rest])
    if args.cmd == "manual":
        return subprocess.call([py, "ans.py"])
    if args.cmd == "eval":
        return subprocess.call([py, "evaluate_kg.py", *args.rest])
    if args.cmd == "app":
        return subprocess.call([py, "-m", "streamlit", "run", "app_streamlit.py", *args.rest])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
