# 第 1 部分：为什么需要排序检索

前面学的是 **Boolean Retrieval**：

- AND
- OR
- NOT

它的问题是：

- 条件严格时，结果可能是 0
- 条件宽松时，结果可能非常多

所以普通用户不适合用。

这一章引入：

> **Ranked Retrieval（排序检索）**

目标是：

> 不只判断匹不匹配，而是判断谁更相关。

所以系统先给每篇文档一个分数：

$score(q,d)$

其中：

- $q$：query
- $d$：document

分数越高，排名越靠前。

------

# 第 2 部分：最基础的打分思路

PPT 先介绍了 **Jaccard 系数**：

$J(A,B)=\frac{|A\cap B|}{|A\cup B|}$

表示两个集合的重合比例。

这里可以把：

- query 看成词集合
- document 看成词集合

但它的问题是：

- 不考虑词频
- 不考虑稀有词的重要性

所以只是一个入门方法。

------

# 第 3 部分：TF（词频）

定义：

$tf_{t,d}$

表示：

> term $t$ 在 document $d$ 中出现的次数。

例如文档：

> car insurance auto insurance

则：

- $tf(car)=1$
- $tf(insurance)=2$
- $tf(auto)=1$

TF 的思想是：

> 查询词在文档里出现越多，通常越相关。

------

# 第 4 部分：为什么 TF 要取 log

不能直接使用原始次数。

因为：

> 出现 10 次，不代表相关性是 1 次的 10 倍。

所以使用 **log-frequency weighting**：

$w_{t,d}=1+\log_{10}(tf_{t,d})$

例如：

- $tf=1 \rightarrow 1$
- $tf=10 \rightarrow 2$
- $tf=1000 \rightarrow 4$

这样权重增长会逐渐变慢。

------

# 第 5 部分：IDF（逆文档频率）

光有 tf 不够。

因为像：

- the
- is
- people

这些词几乎所有文档都有，区分能力很差。

所以加入 **idf**：

$idf_t=\log_{10}\left(\frac{N}{df_t}\right)$

其中：

- $N$：总文档数
- $df_t$：包含 term $t$ 的文档数

含义：

> 出现在越少文档中的词，权重越高。

------

# 第 6 部分：TF-IDF

把 tf 和 idf 结合：

$w_{t,d}=tf \times idf$

完整写法：

$w_{t,d}=(1+\log tf_{t,d})\log\left(\frac{N}{df_t}\right)$

意义是：

> 一个词在当前文档里很多次出现
>  且整个集合里比较少见
>  那它最能代表这篇文档。

这是本章最核心公式。

------

# 第 7 部分：向量空间模型

接下来把文档表示成向量。

假设词表是：

$V=[car,insurance,auto]$

文档：

> car insurance auto insurance

可以写成：

$d=(1,2,1)$

query：

> car insurance

写成：

$q=(1,1,0)$

这样 query 和 document 都能在同一个向量空间里比较。

------

# 第 8 部分：为什么不用欧氏距离

一个自然想法是：

> 谁离 query 更近，谁更相关。

但 PPT 说：

> Euclidean distance is a bad idea 

因为长文档天然向量更长。

即使方向很像，距离也可能很大。

所以不公平。

------

# 第 9 部分：Cosine Similarity

所以改成比较夹角。

公式：

$\cos(q,d)=\frac{q\cdot d}{|q||d|}$

意义：

- 越接近 1：越相关
- 越接近 0：越不相关

所以：

> 用夹角而不是长度比较相似度。

------

# 第 10 部分：长度归一化

为了消除长文档优势，还要做 **归一化**。

向量长度：

$||d||=\sqrt{\sum_i d_i^2}$

归一化后：

$\hat d=\frac{d}{||d||}$

这样所有文档长度都变成 1，更适合公平比较。

------

# 第 11 部分：最终排序流程

整套搜索排序流程就是：

1. query 转成 tf-idf 向量
2. document 转成 tf-idf 向量
3. 计算 cosine similarity
4. 按分数降序排序
5. 返回 top $K$

通常：

$K=10$

这就是搜索引擎最经典的向量空间检索流程。

------

# 第 12 部分：期末重点

你复习重点抓：

## 概念题

- 为什么 Boolean retrieval 不好
- 为什么 tf 要取 log
- 为什么要 idf
- 为什么 cosine 比距离好

## 计算题

- tf
- idf
- tf-idf
- cosine similarity
- top-$K$ 排序







# 例题 1：Jaccard Scoring Example

这是最前面的入门例题。

## 题干

> What is the query-document match score that the Jaccard coefficient computes for each of the two documents below?

Query:
 `ides of march`

Document 1:
 `caesar died in march`

Document 2:
 `the long march`

------

## 解题思路

Jaccard：

$J(A,B)=\frac{|A\cap B|}{|A\cup B|}$

------

## Document 1

### 集合表示

- Query = ${ides, of, march}$
- Doc1 = ${caesar, died, in, march}$

### 交集

${march}$

所以：

$|A\cap B|=1$

### 并集

${ides, of, march, caesar, died, in}$

所以：

$|A\cup B|=6$

### 分数

$J=1/6$

------

## Document 2

- Doc2 = ${the,long,march}$

交集还是：
 ${march}$

并集：

${ides,of,march,the,long}$

大小为 5。

所以：

$J=1/5$

------

## 结论

因为：

$1/5 > 1/6$

所以：

> **Document 2 排名更高**

------

# 例题 2：IDF Example

这个是 **最经典计算题模板**。

## 题干

Suppose:

$N=1,000,000$

求以下词的 idf：

- calpurnia
- animal
- sunday
- fly
- under
- the

其中：

$idf_t=\log_{10}\left(\frac{N}{df_t}\right)$

------

## 题目给的数据

- calpurnia: $df=1$
- animal: $df=100$
- sunday: $df=1000$
- fly: $df=10000$
- under: $df=100000$
- the: $df=1000000$

------

## 结果

- calpurnia: $6$
- animal: $4$
- sunday: $3$
- fly: $2$
- under: $1$
- the: $0$

------

## 解释

它体现：

> **词越稀有，idf 越大**

所以：

> 专有名词通常最重要
>  停用词几乎没价值

------

# 例题 3：3 Documents Cosine Similarity

这是向量空间模型最经典例题。

------

## 题干

比较三本小说的相似度：

- SaS = Sense and Sensibility
- PaP = Pride and Prejudice
- WH = Wuthering Heights

term frequency 如下：

| term      | SaS  | PaP  | WH   |
| --------- | ---- | ---- | ---- |
| affection | 115  | 58   | 20   |
| jealous   | 10   | 7    | 11   |
| gossip    | 2    | 0    | 6    |
| wuthering | 0    | 0    | 38   |

------

## 结果（PPT直接给）

归一化后：

- $\cos(SaS,PaP)\approx 0.94$
- $\cos(SaS,WH)\approx 0.79$
- $\cos(PaP,WH)\approx 0.69$

------

## 解释

因为：

> SaS 和 PaP 在 affection、jealous 上分布最接近

所以最相似。

这是考试很爱出的：

> **给词频表 → 求 cosine similarity → 判断哪两个最相似**

------

# 例题 4：TF-IDF 完整打分题（最重要）

这个就是 **期末大题原型**。

------

## 题干

Document:
 `car insurance auto insurance`

Query:
 `best car insurance`

采用：

> **lnc.ltc**

求最终 score。

------

## PPT 已给关键数据

最后结果：

$Score=0+0+0.27+0.53=0.8$

------

## 为什么这样算

逐词相乘：

### auto

query 中没有

贡献：
 $0$

------

### best

document 中没有

贡献：
 $0$

------

### car

query normalized = $0.52$

doc normalized = $0.52$

乘积：

$0.52\times0.52\approx0.27$

------

### insurance

query normalized = $0.78$

doc normalized = $0.68$

乘积：

$0.78\times0.68\approx0.53$

------

## 总分

$0+0+0.27+0.53=0.8$

------

## 解释

这题本质是：

> **normalized tf-idf 向量点积**

也就是：

$\cos(q,d)$
