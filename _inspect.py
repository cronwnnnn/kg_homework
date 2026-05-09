"""临时调试脚本：抽样查看三元组质量，输出到 _inspect_report.txt。"""
import csv
from collections import Counter


def main():
    rows = list(csv.DictReader(open("output/triples_with_meta.csv", encoding="utf-8-sig")))
    lines: list[str] = []
    lines.append(f"==== 总数: {len(rows)} ====\n")

    lines.append("==== 按来源分布 ====")
    for k, v in Counter(r["source"] for r in rows).most_common():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("==== 按关系类型 Top-30 ====")
    rel_counter = Counter(r["relation"] for r in rows)
    for k, v in rel_counter.most_common(30):
        lines.append(f"  {k}: {v}")
    lines.append("")

    for rel_name, sample_n in [
        ("connected_to", 14),
        ("has_part", 10),
        ("reduces", 12),
        ("improves", 12),
        ("develops", 10),
        ("has_parameter", 10),
        ("uses_method", 10),
        ("manufactures", 8),
        ("has_value", 10),
        ("has_configuration", 8),
        ("instance_of", 8),
    ]:
        sub = [r for r in rows if r["relation"] == rel_name][:sample_n]
        total_in_rel = len([r for r in rows if r["relation"] == rel_name])
        lines.append(f"==== {rel_name} ({total_in_rel} 条, 抽样{len(sub)}) ====")
        for r in sub:
            sentence_short = (r["sentence"] or "")[:80].replace("\n", " ")
            lines.append(f"  {r['head']} -> {r['tail']} | src={r['source']} | trig={r['trigger']} | s={sentence_short}")
        lines.append("")

    with open("_inspect_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("写入 _inspect_report.txt")


if __name__ == "__main__":
    main()
