"""一次性从 ans.EntityLibrary 导出 data/entities_by_type.json。

之后 `run_extract.py --entities-json data/entities_by_type.json` 即可在无 ans 模块
（或不想改 ans）的情况下跑抽取；词典内容仍来源于本次导出时的 ans 定义。

用法：
    uv run python tools/export_entities_by_type_json.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 与 run_extract._EXCLUDED_VOCAB_ETYPES / _EXCLUDED_VOCAB_TERMS 保持一致
_EXCLUDED_VOCAB_ETYPES = {"NUMERIC_VALUE"}
_EXCLUDED_VOCAB_TERMS = {
    "是", "为", "在", "有", "无", "其", "之", "者",
    "部分", "情况", "状态", "方式", "方法", "需要", "需求", "要求",
    "影响", "决定", "可能", "进而", "因此", "通过", "使用", "采用", "利用",
    "来自", "之间", "两者", "三者", "其他", "其它", "上述", "下述",
    "目前", "近年", "随着", "随之", "结果", "结论", "讨论", "分析",
    "研究", "本文", "本章", "本节", "公式", "图表", "表格", "样品",
    "单一", "复杂",
}


def main() -> int:
    from ans import EntityLibrary  # noqa: E402

    out_dir = os.path.join(ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "entities_by_type.json")

    entities_dict = EntityLibrary.get_all_entities()
    data: dict[str, list[str]] = {}
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
            cleaned.append(ww)
        if cleaned:
            data[type_name] = cleaned

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_terms = sum(len(v) for v in data.values())
    print(f"[export] 已写 {out_path}  类型数={len(data)}  词条总数={n_terms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
