import re
import spacy
import ahocorasick

# ========================================================
# 步骤 1：基础抽取层（传统算法主导）
# ========================================================

def split_chinese_sentences(text: str, min_len: int = 8) -> list[str]:
    """按常见中文句末标点切句，避免整段只跑一次依存分析。"""
    text = text.strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[。！？；])\s*", text)
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if len(c) >= min_len:
            out.append(c)
    return out


def normalize_span(s: str) -> str:
    """去掉首尾括号与残留标点，减轻类似 NASA) 的尾噪声。"""
    s = s.strip()
    while s and s[0] in "（([『「":
        s = s[1:].strip()
    while s and s[-1] in "）)]』」、，。；：！？":
        s = s[:-1].strip()
    return s


def keep_triple(head: str, rel: str, tail: str, vocab_set: set[str]) -> bool:
    """过滤过短碎片；头尾至少一端落在领域词典内，便于面向专业图谱。"""
    h, t = normalize_span(head), normalize_span(tail)
    r = rel.strip()
    if len(h) < 2 or len(t) < 2 or len(r) < 1:
        return False
    if h not in vocab_set and t not in vocab_set:
        return False
    return True


class TraditionalExtractor:
    def __init__(self, vocab_list):
        # 1. 初始化 spaCy 中文模型
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except:
            print("请先执行: python -m spacy download zh_core_web_sm")
            raise
        
        # 2. 初始化 Aho-Corasick 自动机 (用于快速字典匹配 NER)
        self.automaton = ahocorasick.Automaton()
        for idx, word in enumerate(vocab_list):
            self.automaton.add_word(word, (idx, word))
        self.automaton.make_automaton()

    def extract_entities_ac(self, text):
        """利用 AC 自动机进行词典匹配 NER"""
        entities = set()
        for end_index, (insert_order, original_value) in self.automaton.iter(text):
            entities.add(original_value)
        return list(entities)

    def extract_svo_relations(self, text):
        """基于 spaCy 依存句法分析提取 SVO 主谓宾作为基础关系"""
        doc = self.nlp(text)
        triples = []
        for token in doc:
            # 找到句子的谓语/根节点
            if token.dep_ == 'ROOT':
                # 寻找主语
                subjects = [w.text for w in token.lefts if w.dep_ in ('nsubj', 'top', 'nsubjpass')]
                # 寻找宾语
                objects = [w.text for w in token.rights if w.dep_ in ('dobj', 'pobj', 'iobj')]
                
                # 如果包含完整的主谓宾，则构成一个传统三元组
                if subjects and objects:
                    for sub in subjects:
                        for obj in objects:
                            triples.append((sub, token.text, obj))
        return triples

# ========================================================
# 整体流水线测试 - 暂不使用 LLM 直接输出为 CSV
# ========================================================
if __name__ == "__main__":
    import csv
    from ans import EntityLibrary # 如果 ans.py 不在同级目录，请确保路径正确

    # 1. 从 ans.py 导入并展平专业词典
    print("正在从 EntityLibrary 加载专业领域实体...")
    entities_dict = EntityLibrary.get_all_entities()
    domain_vocab = []
    for entity_list in entities_dict.values():
        domain_vocab.extend(entity_list)
    
    domain_vocab = list(set(domain_vocab))
    vocab_set = set(domain_vocab)
    print(f"成功加载词典，共包含 {len(domain_vocab)} 个专业实体。")

    # 2. 读取 aftcln.txt 论文内容
    file_path = "aftcln.txt"
    try:
        # 尝试以 utf-8 格式读取
        with open(file_path, "r", encoding="utf-8") as f:
            paper_text = f.read()
    except UnicodeDecodeError:
        # 万一你的 txt 是 GBK 编码（Windows常见），做个容错
        with open(file_path, "r", encoding="gbk") as f:
            paper_text = f.read()

    # 将长文按换行符切分为多段，并过滤空段落或太短的无用段落
    paragraphs = [p.strip() for p in paper_text.split('\n') if len(p.strip()) > 10]
    print(f"论文读取成功，共切分为 {len(paragraphs)} 个有效段落。")

    # 3. 初始化传统处理器
    print("\n正在初始化传统 NLP 模型 (加载 spaCy 模型可能需要几秒钟)...")
    base_nlp = TraditionalExtractor(domain_vocab)
    
    # 4. 第一阶段：逐段 -> 分句 -> SVO；过滤后与去重
    print("\n--- 传统 NLP 逐段分句处理进行中 ---")
    all_base_ents = set()
    all_base_rels: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for i, para in enumerate(paragraphs):
        ents = base_nlp.extract_entities_ac(para)
        all_base_ents.update(ents)

        for sent in split_chinese_sentences(para):
            for sub, pred, obj in base_nlp.extract_svo_relations(sent):
                if not keep_triple(sub, pred, obj, vocab_set):
                    continue
                h, t = normalize_span(sub), normalize_span(obj)
                r = pred.strip()
                key = (h, r, t)
                if key in seen:
                    continue
                seen.add(key)
                all_base_rels.append(key)

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1} / {len(paragraphs)} 段...")

    print(f"\n处理完成！总共在全文中命中专业实体 {len(all_base_ents)} 个，提取出去重后主谓宾三元组 {len(all_base_rels)} 条。")

    # 5. 导出为 CSV 文档
    output_csv = "knowledge_graph.csv"
    with open(output_csv, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["头实体", "关系", "尾实体"]) # 写表头
        for triple in all_base_rels:
            writer.writerow(triple)

    print(f"\n成功！所有三元组已保存至 {output_csv} 文件中。")