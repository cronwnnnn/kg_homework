==== 银标说明（与 evaluate_kg.py 配套） ====

silver_entities.txt
    论文中真实出现的领域实体清单（与 run_extract 一致的过滤后词表）。
silver_triples.csv         <- evaluate_kg.py 默认评估目标
    严格银标：基于触发词派生 30+ 种真实关系（improves/reduces/has_part/...），
    未命中触发词的相邻实体对兜底为 co_occurs_with。
    适用于关系级别 P/R/F1 评估。
silver_triples_loose.csv
    宽松银标：所有同句相邻实体对均记 co_occurs_with，仅做实体覆盖评估。
    命令：evaluate_kg.py --gold gold/silver_triples_loose.csv --loose

注：本银标为自动派生，非纯人工金标。若课程要求纯人工金标，
请以 silver_triples.csv 为模板逐行复核为 gold_triples_labeled.csv。
