"""三元组、提及、关系本体的数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Mention:
    """实体提及：一个具体在文本中出现的实体片段。"""

    text: str
    start: int
    end: int
    etype: str = "Entity"

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class Triple:
    """带元信息的三元组。

    relation 为归一化后的关系类型；trigger 为原始触发词/谓词；
    source 标记产生该三元组的子算法；score 为来源给出的置信度。
    """

    head: str
    relation: str
    tail: str
    trigger: str = ""
    source: str = "rule"
    score: float = 1.0
    chapter: str = ""
    sentence: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


class RelationOntology:
    """统一关系本体：触发词 → 关系类型 的映射表。

    在抽取时，命中关键字即可映射；在归一化时，spaCy/LLM 给出的自由谓词也走这张表。
    顺序敏感：长触发词必须排在短触发词之前，避免被前缀吃掉（例如"提升"应早于"升"）。
    """

    TRIGGER_TABLE: dict[str, list[str]] = {
        "has_part": ["由其组成", "组成包括", "由…组成", "由…构成", "包括", "包含", "含有", "由..组成", "构成", "组成", "分为", "分成"],
        "is_a": ["是一种", "是一类", "属于", "称之为", "称为"],
        "used_for": ["应用于", "适用于", "用于", "用作", "使用于", "服务于", "面向"],
        "applied_to": ["应用到", "运用到", "推广至", "推广到"],
        # 注：删除短歧义触发词（"扩大/增大/提高/提升 / 减小/减少 / 引起 / 进而(使)" 等），
        # 它们在第四章评估中 P=0~0.05，几乎全是 FP。保留长触发词 + 高确定性短词。
        "improves": ["显著提升", "大幅提升", "进一步提升", "改善", "增强", "强化"],
        "reduces": ["大幅降低", "显著降低", "进一步降低", "削弱", "抑制", "缓解", "缩小", "缩减"],
        "causes": ["导致", "造成", "引发", "致使"],
        "leads_to": ["使得"],
        "controls": ["控制", "调控", "调节", "调整", "操纵"],
        "driven_by": ["由…驱动", "由其驱动"],
        "drives": ["驱动", "推动"],
        "actuated_by": ["由作动器驱动"],
        "manufactures": ["研制", "制造", "生产"],
        # 删 "设计"（独立 2 字 P=0.03，误触发 develops"设计参数/设计点"）；保留长触发词
        "develops": ["研发", "开发", "提出了", "提出", "发展和完善", "发展", "完善", "设计了", "设计出"],
        "originates_from": ["起源于", "源于", "始于", "始建于", "诞生于"],
        # 数值比较类：在 numeric_extractor 里专门用数字模板匹配；trigger 这里只保留长触发词，
        # 避免"大于/小于/等于"误触发到非数值实体对（评估显示 P=0）。
        "greater_than": ["远大于"],
        "less_than": ["远小于", "不超过"],
        "equals_to": ["相当于", "约等于", "近似为"],
        "depends_on": ["取决于", "依赖于", "决定于"],
        "affects": ["影响", "决定", "关系到", "关乎", "作用于"],
        "uses_method": ["采用", "采取", "借助", "通过", "运用", "利用"],
        "verifies": ["验证", "检验", "证明", "证实"],
        "implements": ["实现", "达成"],
        "satisfies": ["满足", "符合"],
        "needs": ["需要", "需求"],
        "solves": ["解决", "克服"],
        "located_at": ["位于", "处于", "处在", "安装于", "安装在", "设置于", "布置于"],
        "connected_to": ["连接", "对接", "连接到", "连接至", "铰接", "固连"],
        "combines_with": ["结合", "配合", "联合", "兼具", "兼作"],
        "transforms_to": ["转变为", "切换为", "变体为", "转化为", "变成"],
        "generates": ["产生", "生成", "形成"],
        "provides": ["提供", "给出"],
    }

    # 列入"短/弱触发词"：命中需要 (a) 紧邻头/尾实体 (b) 句中无强触发词替代
    WEAK_TRIGGERS: set[str] = {
        "为", "是", "属于",
        "影响", "决定",
        "构成", "形成", "产生", "生成",
        "用于", "包括", "包含", "采用", "使用", "通过",
    }

    INVERSE_RELATIONS = {
        "has_part": "part_of",
        "is_a": "instance_of",
        "developed_by": "develops",
        "depends_on": "enables",
        "controls": "controlled_by",
        "driven_by": "drives",
        "located_at": "contains",
        "connected_to": "connected_to",
        "transforms_to": "transformed_from",
        "generates": "generated_by",
    }

    @classmethod
    def all_triggers(cls) -> Iterable[tuple[str, str]]:
        """yield (trigger, relation_type)，长触发词优先。"""
        flat: list[tuple[str, str]] = []
        for rel, words in cls.TRIGGER_TABLE.items():
            for w in words:
                flat.append((w, rel))
        flat.sort(key=lambda x: -len(x[0]))
        return flat

    # 关系-实体类型一致性约束：head/tail 中不允许出现某些 mention etype。
    # 例如 is_a 关系的尾部不应是数值。
    RELATION_TYPE_CONSTRAINTS: dict[str, dict[str, set[str]]] = {
        "is_a": {"tail_forbid": {"numeric"}},
        "instance_of": {"tail_forbid": {"numeric"}},
        "located_at": {"tail_forbid": {"numeric"}},
        "manufactures": {"tail_forbid": {"numeric"}},
        "develops": {"tail_forbid": {"numeric"}},
        "transforms_to": {"tail_forbid": {"numeric"}},
        "has_part": {"tail_forbid": {"numeric"}},
        "uses_method": {"tail_forbid": {"numeric"}},
    }
