"""项目主入口：子命令转发到各脚本。"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="单-双折叠翼变体飞行器知识图谱构建系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="从 aftcln.txt 自动抽取（run_extract）")
    p_ext.add_argument("rest", nargs=argparse.REMAINDER, help="透传给 run_extract.py")

    sub.add_parser("manual", help="打印 ans.py 领域词典统计（EntityLibrary，不生成图谱）")

    p_silver = sub.add_parser("silver", help="导出银标评估集 gold/silver_*.csv")
    p_silver.add_argument("rest", nargs=argparse.REMAINDER)

    p_eval = sub.add_parser("eval", help="三元组/实体 P、R、F1（evaluate_kg）")
    p_eval.add_argument("rest", nargs=argparse.REMAINDER)

    p_app = sub.add_parser("app", help="Streamlit 图谱浏览（需 pip install -e \".[app]\"）")
    p_app.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.cmd == "extract":
        return subprocess.call([sys.executable, "run_extract.py", *args.rest])
    if args.cmd == "manual":
        return subprocess.call([sys.executable, "ans.py"])
    if args.cmd == "silver":
        return subprocess.call([sys.executable, "tools/export_silver_gold.py", *args.rest])
    if args.cmd == "eval":
        return subprocess.call([sys.executable, "evaluate_kg.py", *args.rest])
    if args.cmd == "app":
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "app_streamlit.py", *args.rest]
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
