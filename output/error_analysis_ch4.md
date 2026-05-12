# 第四章评估 · 错误分析报告

- pred: `output/triples_with_meta.csv`
- gold: `gold/gold_triples.csv`
- chapter: `第4章`  include_global=True

## 总览

- pred 三元组: **1249**，pred 实体: **668**
- gold 三元组: **442**，gold 实体: **478**
- 实体重叠: **183** (占 gold 的 38.3%)
- TP / FP / FN: **103 / 1146 / 339**

## FN（漏抽）共 339 条

### 按错因分类

| 类别 | 数量 | 占比 |
|------|-----:|-----:|
| C. tail 识别 / head 漏识 | 116 | 34.2% |
| D. 两端实体都没识别 | 103 | 30.4% |
| A. 两端实体都识别了 → 关系/规则缺失 | 70 | 20.6% |
| B. head 识别 / tail 漏识 | 50 | 14.7% |

### 各类别样例（最多 6 条）

**C. tail 识别 / head 漏识**

- `25%弦长` —[instance_of]→ `设计参数`
- `45%弦长` —[instance_of]→ `设计参数`
- `Ahmed` —[instance_of]→ `人物`
- `CFD` —[instance_of]→ `技术`
- `CFD时间步长` —[has_value]→ `0.005秒`
- `CFD模型` —[has_part]→ `嵌套网格`

**A. 两端实体都识别了 → 关系/规则缺失**

- `Addoms` —[instance_of]→ `人物`
- `Laitone` —[instance_of]→ `人物`
- `上下机翼` —[has_part]→ `桁架式结构`
- `上下机翼间距` —[increases]→ `升力线斜率`
- `上下机翼间距` —[increases]→ `奥斯瓦尔德因子`
- `上下机翼间距` —[increases]→ `最大升力系数`

**D. 两端实体都没识别**

- `Boeing106` —[instance_of]→ `翼型`
- `NACA0012` —[instance_of]→ `翼型`
- `NACA2412` —[instance_of]→ `翼型`
- `三维双翼计算` —[uses_method]→ `三维重叠网格`
- `上下机翼相互干扰` —[causes]→ `升力线斜率损失`
- `上下机翼相互干扰` —[causes]→ `最大升力系数损失`

**B. head 识别 / tail 漏识**

- `NASA` —[develops]→ `连接翼布局`
- `U型槽` —[used_for]→ `增强抗扭性能`
- `上机翼` —[located_at]→ `下机翼前方`
- `上机翼` —[located_at]→ `下机翼后方`
- `上机翼` —[located_at]→ `下机翼斜前方`
- `上机翼` —[located_at]→ `下机翼斜后方`

### FN 按关系类型分布（Top-15）

| relation | FN |
|----------|---:|
| instance_of | 64 |
| has_value | 51 |
| has_part | 40 |
| uses_method | 24 |
| causes | 24 |
| located_at | 23 |
| used_for | 20 |
| reduces | 16 |
| affects | 15 |
| develops | 10 |
| generates | 9 |
| increases | 7 |
| improves | 7 |
| equivalent_to | 7 |
| defined_as | 6 |

## FP（误抽）共 1146 条

### 按错因分类

| 类别 | 数量 | 占比 |
|------|-----:|-----:|
| D. 一端是 gold 实体 / 一端误抓 | 635 | 55.4% |
| E. 两端都不在 gold → 完全误抓 | 321 | 28.0% |
| C. 两端都是 gold 实体但关系不对 | 174 | 15.2% |
| A. head 是数值/年份 → 抽取边界错 | 15 | 1.3% |
| B. has_part 的 tail 是数值 → 应该是 has_value | 1 | 0.1% |

### 各类别样例（最多 6 条）

**A. head 是数值/年份 → 抽取边界错**

- `0.7m` —[has_part]→ `上翼`
- `0.7m` —[has_part]→ `下翼`
- `0.7m` —[has_part]→ `小于`
- `0.7m` —[has_part]→ `弦长`
- `0.7m` —[less_than]→ `下翼`
- `0.7m` —[less_than]→ `弦长`

**D. 一端是 gold 实体 / 一端误抓**

- `10自由度动力学模型` —[instance_of]→ `概念`
- `1倍弦长` —[improves]→ `时机翼`
- `2自由度动力学模型` —[instance_of]→ `概念`
- `ACADO` —[instance_of]→ `技术`
- `Aquila` —[instance_of]→ `飞行器`
- `B-1` —[instance_of]→ `飞行器`

**E. 两端都不在 gold → 完全误抓**

- `DARPA` —[instance_of]→ `组织机构`
- `Facebook公司` —[instance_of]→ `组织机构`
- `NACA2412翼型` —[uses_method]→ `数值计算`
- `一部分` —[equals_to]→ `重力`
- `一部分` —[reduces]→ `损失`
- `三维展` —[used_for]→ `飞行器总体设计`

**C. 两端都是 gold 实体但关系不对**

- `U型槽` —[instance_of]→ `结构部件`
- `Z字形折叠翼` —[instance_of]→ `机翼构型`
- `上下机翼` —[greater_than]→ `升力`
- `上下机翼` —[improves]→ `最大升力系数`
- `上下机翼` —[leads_to]→ `展向升力分布`
- `上下机翼` —[leads_to]→ `最大升力系数`

**B. has_part 的 tail 是数值 → 应该是 has_value**

- `来流攻角` —[has_part]→ `2度`

### FP 按关系类型分布（Top-15）

| relation | FP |
|----------|---:|
| instance_of | 253 |
| discussed_in | 95 |
| reduces | 90 |
| located_at | 74 |
| greater_than | 61 |
| develops | 55 |
| improves | 52 |
| leads_to | 49 |
| has_part | 43 |
| needs | 43 |
| equals_to | 42 |
| causes | 34 |
| less_than | 32 |
| uses_method | 29 |
| transforms_to | 24 |

## gold 中未被 pred 识别的实体共 295 个

- 其中数值型（建议忽略 / schema 调整）: **26** 个
- 非数值型（建议扩充 NER 词典）: **269** 个

### 非数值型未识别 gold 实体（最多 60 个）

- `-3到3倍弦长`
- `0.25到3倍弦长`
- `10度每秒`
- `25%弦长`
- `45%弦长`
- `46%机翼展长`
- `Ahmed`
- `Boeing106`
- `CFD`
- `CFD时间步长`
- `CFD模型`
- `CFD计算`
- `NACA0012`
- `NACA2412`
- `NextGenAeronautics`
- `Prosnak`
- `URANS方法`
- `三维双翼`
- `三维双翼计算`
- `三维重叠网格`
- `上下机翼总弦长`
- `上下机翼总面积`
- `上下机翼相互干扰`
- `上下机翼距离过近`
- `上下翼平均间距`
- `上下翼间距推荐`
- `上下翼间距范围`
- `上下间距除以平均弦长`
- `上机翼上反`
- `上机翼下表面`
- `上机翼在前`
- `上机翼在后`
- `上机翼实际攻角增加`
- `上机翼实际攻角降低`
- `上机翼带一定上反角`
- `上机翼弦长大于下机翼`
- `上机翼弦长小于下机翼`
- `上机翼弦长扩大`
- `上机翼弦长缩小`
- `上机翼弯度变化`
- `上机翼弯度大于下机翼`
- `上机翼弯度小于下机翼`
- `上机翼斜前方`
- `上机翼斜后方`
- `上机翼正上方`
- `上机翼运动`
- `上翼弦长扩大`
- `上翼弦长缩小`
- `下机翼上表面`
- `下机翼前方`
- `下机翼后方`
- `下机翼在前`
- `下机翼实际攻角增加`
- `下机翼实际攻角降低`
- `下机翼斜前方`
- `下机翼斜后方`
- `下机翼正上方`
- `不对称阻力产生扭转力矩`
- `两根主梁之间`
- `主油箱`
- … 还有 209 个未列出

## 结论 & 建议

- FN 主因: **C. tail 识别 / head 漏识** (34.2%)
- FP 主因: **D. 一端是 gold 实体 / 一端误抓** (55.4%)

**改进策略推荐**：

- 如果 FN 主因是 D（两端都没识别）→ 优先扩充 NER 词典
- 如果 FN 主因是 A（实体识别了但关系缺失）→ 增加关系触发词 / 修关系抽取规则
- 如果 FP 主因是 A（head 是数值/年份）→ 在抽取后处理过滤纯数值/年份主语
- 如果 FP 主因是 C（关系错）→ 修关系方向 / 模板
- gold 是否需要改？仅以下情况合法：
  1. 发现事实错误（与论文原文矛盾）
  2. 关系命名不规范（已通过归一化自动处理）
  3. 想明确 schema 范围（如不抽事件状态短语）→ 这是设计决策，不是为涨分
