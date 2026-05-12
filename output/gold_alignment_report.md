# Gold 对齐报告

- 输入: `D:\xiazai\know_graphic\code\my-project\gold\gold_triples.csv` (444 条)
- 输出: `D:\xiazai\know_graphic\code\my-project\gold\gold_triples_aligned.csv` (432 条)
- 改写条数: **38**
- 未改写: 404
- 改写后合并/重复（被丢弃）: 12

## 全部改动

- `(平均前后间距, instance_of, 设计参数)` → `(前后间距, instance_of, 设计参数)`  *head: '平均前后间距' → '前后间距'*
- `(平均上下间距, instance_of, 设计参数)` → `(上下间距, instance_of, 设计参数)`  *head: '平均上下间距' → '上下间距'*
- `(失速同步性, instance_of, 气动概念)` → `(失速, instance_of, 气动概念)`  *head: '失速同步性' → '失速'*
- `(弦向压力分布, instance_of, 气动概念)` → `(压力分布, instance_of, 气动概念)`  *head: '弦向压力分布' → '压力分布'*
- `(双翼整体弯度, instance_of, 气动概念)` → `(弯度, instance_of, 气动概念)`  *head: '双翼整体弯度' → '弯度'*
- `(URANS方法, instance_of, 技术)` → `(RANS方法, instance_of, 技术)`  *head: 'URANS方法' → 'RANS方法'*
- `(二维重叠网格, instance_of, 技术)` → `(重叠网格, instance_of, 技术)`  *head: '二维重叠网格' → '重叠网格'*
- `(三维重叠网格, instance_of, 技术)` → `(重叠网格, instance_of, 技术)`  *head: '三维重叠网格' → '重叠网格'*
- `(早期双翼机, has_part, 支柱)` → `(双翼机, has_part, 支柱)`  *head: '早期双翼机' → '双翼机'*
- `(连杆与活动机翼连接点, located_at, 45%弦长)` → `(活动机翼, located_at, 45%弦长)`  *head: '连杆与活动机翼连接点' → '活动机翼'*
- `(展向升力中心变体后, located_at, 50%机翼展长)` → `(变体, located_at, 机翼)`  *head: '展向升力中心变体后' → '变体'; tail: '50%机翼展长' → '机翼'*
- `(后失速机翼, leads_to, 失速提前)` → `(机翼, leads_to, 失速提前)`  *head: '后失速机翼' → '机翼'*
- `(机身, leads_to, 连杆变形)` → `(机身, leads_to, 连杆)`  *tail: '连杆变形' → '连杆'*
- `(翼尖负担转角, reduces, 翼尖升力系数)` → `(负担转角, reduces, 翼尖)`  *head: '翼尖负担转角' → '负担转角'; tail: '翼尖升力系数' → '翼尖'*
- `(双翼构型, reduces, 起降翼展)` → `(双翼构型, reduces, 起降)`  *tail: '起降翼展' → '起降'*
- `(单翼构型, improves, 巡航升阻比)` → `(单翼构型, improves, 巡航)`  *tail: '巡航升阻比' → '巡航'*
- `(公转角度_单翼状态, has_value, 0度)` → `(公转角, has_value, 0度)`  *head: '公转角度_单翼状态' → '公转角'*
- `(公转角度_双翼状态, has_value, 157度)` → `(公转角度, has_value, 157度)`  *head: '公转角度_双翼状态' → '公转角度'*
- `(公转角度_最高点, has_value, 90度)` → `(公转角度, has_value, 90度)`  *head: '公转角度_最高点' → '公转角度'*
- `(副翼面积占比_最小, has_value, 15%)` → `(副翼面积占比, has_value, 15%)`  *head: '副翼面积占比_最小' → '副翼面积占比'*
- `(副翼面积占比_最大, has_value, 25%)` → `(副翼面积占比, has_value, 25%)`  *head: '副翼面积占比_最大' → '副翼面积占比'*
- `(机身和固定机翼重量, has_value, 10000kg)` → `(固定机翼, has_value, 10000kg)`  *head: '机身和固定机翼重量' → '固定机翼'*
- `(每个活动机翼重量, has_value, 800kg)` → `(活动机翼, has_value, 800kg)`  *head: '每个活动机翼重量' → '活动机翼'*
- `(不对称阻力产生扭转力矩, has_value, 1150N·m)` → `(阻力, has_value, 1150N·m)`  *head: '不对称阻力产生扭转力矩' → '阻力'*
- `(副翼偏转产生俯仰力矩, has_value, 3200N·m)` → `(副翼偏转, has_value, 3200N·m)`  *head: '副翼偏转产生俯仰力矩' → '副翼偏转'*
- `(连杆重量, has_value, 120kg)` → `(连杆, has_value, 120kg)`  *head: '连杆重量' → '连杆'*
- `(连杆最大弯矩, has_value, 7000N·m)` → `(连杆, has_value, 7000N·m)`  *head: '连杆最大弯矩' → '连杆'*
- `(连杆最大扭矩, has_value, 3200N·m)` → `(连杆, has_value, 3200N·m)`  *head: '连杆最大扭矩' → '连杆'*
- `(铰链重量, has_value, 600克)` → `(铰链, has_value, 600克)`  *head: '铰链重量' → '铰链'*
- `(副翼, used_for, 控制活动机翼公转)` → `(副翼, used_for, 机翼)`  *tail: '控制活动机翼公转' → '机翼'*
- `(副翼, used_for, 控制活动机翼自转)` → `(副翼, used_for, 机翼)`  *tail: '控制活动机翼自转' → '机翼'*
- `(副翼, used_for, 卸载活动机翼升力)` → `(副翼, used_for, 机翼)`  *tail: '卸载活动机翼升力' → '机翼'*
- `(副翼差动, used_for, 控制活动机翼自转)` → `(副翼, used_for, 机翼)`  *head: '副翼差动' → '副翼'; tail: '控制活动机翼自转' → '机翼'*
- `(普朗特双翼理论, applied_to, 串列双翼布局)` → `(双翼理论, applied_to, 串列双翼布局)`  *head: '普朗特双翼理论' → '双翼理论'*
- `(陈海昕课题组, develops, 单-双折叠翼变体飞行器)` → `(课题组, develops, 变体飞行器)`  *head: '陈海昕课题组' → '课题组'; tail: '单-双折叠翼变体飞行器' → '变体飞行器'*
- `(单-双折叠翼变体, solves, 大展弦比飞机起降与巡航性能的矛盾)` → `(单-双折叠翼变体, solves, 展弦比)`  *tail: '大展弦比飞机起降与巡航性能的矛盾' → '展弦比'*
- `(双翼气动等效原则, develops, 双翼气动设计)` → `(等效原则, develops, 双翼)`  *head: '双翼气动等效原则' → '等效原则'; tail: '双翼气动设计' → '双翼'*
- `(通用原子, develops, 前线机场无人机)` → `(通用原子, develops, 前线机场)`  *tail: '前线机场无人机' → '前线机场'*
