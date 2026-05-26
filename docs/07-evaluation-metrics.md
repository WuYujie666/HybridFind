# 第八章 Evaluation（评价）主要讲什么

这个 PPT 的核心问题是：

**如何判断一个搜索系统是否“搜得好”？**

也就是：

> 如何量化搜索结果质量（relevance / ranking quality）

---

# 1. Evaluation 的基本框架

一个标准 benchmark 由三部分组成：

1. **Document Collection**
   - 文档集合
   - 例如 500 万商品

2. **Query Set**
   - 测试查询集合
   - 例如 5 万条真实 query

3. **Relevance Judgments**
   - 相关性标注
   - 判断 document 对 query 是否相关

即：

$$(q,d)\rightarrow rel$$

其中：

- $q$ = query
- $d$ = document
- $rel$ = relevance label

---

# 2. 最基础评价指标：Precision / Recall

这是最基础的离线评价指标。

## Precision（查准率）

在检索出的文档里，有多少是真的相关。

$$P=\frac{TP}{TP+FP}$$

其中：

- $TP$：relevant 且 retrieved
- $FP$：not relevant 但 retrieved

---

## Recall（查全率）

所有相关文档中，有多少被找回。

$$R=\frac{TP}{TP+FN}$$

其中：

- $FN$：relevant 但没有被检索出来

---

# 3. 排序评价：P@K

用户通常只看前 $K$ 个结果。

所以定义：

$$P@K=\frac{\text{Top K 中 relevant 文档数}}{K}$$

例如：

- top 5 里有 4 个相关

则：

$$P@5=\frac{4}{5}=0.8$$

---

# 4. MAP（考试重点）

MAP = Mean Average Precision

适用于：

> 一个 query 有多个 relevant documents

---

## 单个 query：AP

先求每个相关文档出现位置上的 precision，再平均：

$$AP(q)=\frac{1}{R}\sum_{k\in RelPos}P@k$$

其中：

- $R$ = relevant 文档总数
- $RelPos$ = relevant doc 出现的位置集合

---

## 多 query：MAP

对所有 query 的 AP 再平均：

$$MAP=\frac{1}{|Q|}\sum_{q\in Q}AP(q)$$

含义：

> 衡量系统整体把相关文档排前面的能力

---

# 5. 多级相关性：DCG

Web 搜索更常用多级相关性：

- 0 = 不相关
- 1 = 一般相关
- 2 = 很相关
- 3 = 高度相关

所以使用 DCG。

## DCG 定义

$$DCG_p=rel_1+\sum_{i=2}^{p}\frac{rel_i}{\log_2 i}$$

含义：

- 越靠前，贡献越大
- 越靠后，被 discount（折损）

---

# 6. NDCG（最重要）

不同 query 的 relevant 数量不同，DCG 不能直接比较。

所以归一化：

$$NDCG_p=\frac{DCG_p}{IDCG_p}$$

其中：

- $IDCG_p$ = ideal ranking 的 DCG
- 即最优排序的 DCG

所以：

$$0\le NDCG\le 1$$

越接近 1 越好。

---

# 7. MRR（只有一个答案时）

如果用户只想找一个正确答案：

- navigational query
- fact query
- known-item query

使用：

$$RR=\frac{1}{rank}$$

多个 query 平均：

$$MRR=\frac{1}{|Q|}\sum RR(q)$$

---

# 8. 用户点击评价（工业界重点）

后半部分 PPT 在讲：

> 用真实用户点击行为评估排序质量

优点：

- 不需要人工标注
- 数据量极大
- 更贴近真实用户

---

## 位置偏差（Position Bias）

高位结果天然更容易被点击。

即使质量一样：

> 用户也更容易点前面的结果

所以：

> clicks are informative but biased

---

# 9. Interleaving（比较两个排序）

用于比较两个排序算法 $A$ 和 $B$

流程：

1. 交替合并 A/B 排序结果
2. 去重
3. 展示给用户
4. 统计点击来自哪一方

如果 A 点击更多：

$$A>B$$

说明 A 排序更好。

---

# 10. A/B Testing

真实搜索引擎上线前常用。

做法：

- 旧系统：99.9%
- 新系统：0.1%

比较：

- CTR
- dwell time
- conversion
- revenue

---

# 一句话总结

这章主要讲：

> 如何使用离线指标（Precision / MAP / NDCG）和在线用户行为（clicks / A-B test）评价搜索系统质量