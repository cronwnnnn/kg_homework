import spacy
import ahocorasick
from openai import OpenAI

# ========================================================
# 步骤 1：基础抽取层（传统算法主导）
# ========================================================

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
# 步骤 2：LLM 优化与泛化层
# ========================================================

class LLMEnhancer:
    def __init__(self, api_key, base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-3.5-turbo" # 替换为你实际使用的模型名称

    def align_entities(self, entity_list):
        """LLM 消歧与实体对齐"""
        prompt = (
            "你是一个知识图谱专家。以下是传统NLP提取出的实体列表，其中可能存在同指代异名的现象（例如'张三'和'小张'）。\n"
            f"实体列表: {entity_list}\n"
            "请将指向同一对象的实体进行对齐聚合，输出一个 JSON，格式为 {'主体名': ['别名1', '别名2']}，不要输出多余解释。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

    def infer_complex_relations(self, text, base_entities, base_triples):
        """LLM 复杂关系推理 (基于基础提取清单)"""
        prompt = (
            "你是一个图谱三元组抽取引擎。我将提供一段文本以及前置算法提取的基础实体和部分粗糙关系。\n"
            f"原文本：{text}\n"
            f"基础算法提取的实体：{base_entities}\n"
            f"基础算法提取的关系：{base_triples}\n\n"
            "任务：\n"
            "1. 修正基础关系中的语法错误或噪音。\n"
            "2. 继续从文本中挖掘深层关系。\n"
            "3. 只输出 CSV 格式的最终三元组，每行格式为：头实体,关系,尾实体。不输出额外文本。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content


# ========================================================
# 整体流水线测试
# ========================================================
if __name__ == "__main__":
    # 配置你的专业词典
    domain_vocab = ["编译器", "词法分析器", "语法树", "张三", "小张", "LLVM", "前端", "后端"]
    text_sample = "张三是这个项目的负责人，小张指出，现代编译器包含了词法分析器和语法树。它通常分为前端和后端。"

    # 1. 初始化两层处理器
    base_nlp = TraditionalExtractor(domain_vocab)
    
    # 填入你的大模型 API KEY (如 DeepSeek、Kimi 等都可以)
    llm_enhancer = LLMEnhancer(api_key="your_api_key_here")

    # 2. 第一阶段：传统 NLP 极速处理
    print("--- 第一阶段：传统 NLP 处理 ---")
    base_ents = base_nlp.extract_entities_ac(text_sample)
    base_rels = base_nlp.extract_svo_relations(text_sample)
    print("基础实体命中:", base_ents)
    print("基础 SVO 提取:", base_rels)

    # 3. 第二阶段：LLM 介入，实体对齐与深层推理
    print("\n--- 第二阶段：LLM 优化处理 ---")
    
    # 实体对齐测试 (可定期对齐全库实体，而非逐句)
    # aligned_ents = llm_enhancer.align_entities(base_ents)
    # print("LLM 实体对齐结果:", aligned_ents)

    # 复合关系推理测试
    final_triples = llm_enhancer.infer_complex_relations(text_sample, base_ents, base_rels)
    print("LLM 最终生成图谱关系:\n", final_triples)