silver_entities.txt: 与 run_extract 一致的过滤后领域词表中，在 aftcln.txt 出现的实体，一行一个。
silver_triples.csv: 每句 HybridNER 命中实体之间 co_occurs_with，头尾均为正文子串，全局去重后截断。
用于自动抽取算法的可复现评估；若课程要求「纯人工金标」，请另建标注文件。
