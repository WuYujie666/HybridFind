# 第九&十一章 Probabilistic IR（概率检索）主要讲什么

这个 PPT 的主线非常清晰：

> 从 Boolean Retrieval 过渡到 **Ranked Retrieval**
> 再用 **概率模型解释为什么 BM25 合理**
> 最后落到现代最重要的排序函数 **BM25 / BM25F**。 :contentReference[oaicite:0]{index=0}

---

# 1. 为什么 Boolean Search 不够好

课件开头先讲 Boolean 检索的问题：

> **要么太少，要么太多（feast or famine）** :contentReference[oaicite:1]{index=1}

比如：

- AND 太严格 → 0 个结果
- OR 太宽松 → 上万结果

所以现代 IR 的核心变成：

> **不要只判断 match / not match**
> 而是要给每个文档一个 score，然后排序

即：

$$score(q,d)$$

---

# 2. 概率排序原则 PRP（整章核心）

这一章最核心理论是：

> **Probability Ranking Principle (PRP)** :contentReference[oaicite:2]{index=2}

核心思想：

> 按“文档相关概率”从高到低排序，就是最优排序

即：

$$P(R=1\mid d,q)$$

其中：

- $R=1$：文档 relevant
- $d$：document
- $q$：query

所以排序目标变成：

> 谁更可能 relevant，谁排前面

---

# 3. Binary Independence Model（BIM）

这是最经典的概率检索模型。 :contentReference[oaicite:3]{index=3}

名字拆开理解：

- **Binary**
  - 只看 term 在不在
  - 不看出现几次

- **Independence**
  - 假设各 term 相互独立

文档表示成：

$$x_i\in\{0,1\}$$

其中：

- $x_i=1$：term $i$ 出现
- $x_i=0$：没出现

---

## BIM 的最终排序形式：RSV
课件推导后，最后都化成 Retrieval Status Value：

$$RSV=\sum_{i:x_i=q_i=1}c_i$$

其中 term weight 为：

$$c_i=\log\frac{p_i(1-r_i)}{r_i(1-p_i)}$$

这里：

- $p_i$：relevant doc 中 term 出现概率
- $r_i$：non-relevant doc 中出现概率

本质就是：

> **某词越能区分 relevant / non-relevant，权重越高**

---

# 4. BIM 为什么会推到 IDF（考试重点）

这是这章一个非常重要的结论。 :contentReference[oaicite:4]{index=4}

在合理近似下：

$$c_i\approx \log\frac{N}{df_i}$$

也就是：

> **IDF 其实是概率模型自然推出来的**

所以你之前学的 tf-idf 里的 idf：

$$idf=\log\frac{N}{df}$$

不是拍脑袋想出来的。

它来自：

> **概率 relevance discrimination**

这是考试很爱问的理论点。

---

# 5. Relevance Feedback

如果知道部分 relevant docs，可以重新估计概率。

即更新：

$$p_i=\frac{|V_i|}{|V|}$$

其中：

- $V$：反馈得到的 relevant set
- $V_i$：其中包含 term $i$ 的文档

于是可以：

> 检索 → 用户反馈 → 更新权重 → 再检索 :contentReference[oaicite:5]{index=5}

这就是 probabilistic relevance feedback。

---

# 6. 为什么 BIM 不够：它不看 TF

BIM 只看：

> term 是否出现

不看：

> 出现多少次

所以它不适合现代全文搜索。 :contentReference[oaicite:6]{index=6}

例如：

- doc1: machine 出现 1 次
- doc2: machine 出现 20 次

BIM 看起来一样。

这显然不合理。

所以课件进入：

> **BM25**

---

# 7. BM25（最最重要）

这一章真正考试核心就是 BM25。 :contentReference[oaicite:7]{index=7}

BM25 的核心思想：

> 结合 **IDF + TF饱和 + 文档长度归一化**

---

## BM25 主公式（必须会）

:contentReference[oaicite:8]{index=8}

---

## 各部分含义

### 1）IDF 部分
$$\log\frac{N}{df_i}$$

词越稀有，越重要。

---

### 2）TF Saturation（高频不无限涨）
核心是：

$$\frac{tf}{k_1+tf}$$

这是饱和函数。

特点：

> tf 增大时分数继续涨，但越来越慢

避免：
- 一个词出现 1000 次把分数拉爆

---

### 3）长度归一化
文档长度因子：

$$B=(1-b)+b\frac{dl}{avdl}$$

其中：

- $dl$：当前文档长度
- $avdl$：平均长度
- $b\in[0,1]$

作用：

> 长文档天然 tf 更大，需要适当惩罚

---

# 8. BM25 两个超参数（必考）

## $k_1$
控制 tf 饱和速度：

- 大：更接近 raw tf
- 小：更快饱和

通常：

$$k_1\in[1.2,2.0]$$

---

## $b$
控制长度归一化：

- $b=0$：不做长度归一化
- $b=1$：完全归一化

通常：

$$b=0.75$$

:contentReference[oaicite:9]{index=9}

---

# 9. 为什么 BM25 比 tf-idf 更好

课件专门举了例子。 :contentReference[oaicite:10]{index=10}

对于 query：

$$[machine\ learning]$$

两个文档：

- doc1: learning 很多，machine 很少
- doc2: 两个词都比较平衡

tf-idf 容易偏向 doc1。

但 BM25 因为 tf saturation：

> 不会让某一个词出现过多次“刷分”

所以更符合真实 relevance。

---

# 10. BM25F（多字段搜索）

最后升级到：

> **BM25F = BM25 with fields / zones** :contentReference[oaicite:11]{index=11}

适合网页：

- title
- body
- anchor
- url

不同区域权重不同。

先做加权 tf：

$$\tilde{tf_i}=\sum_z v_ztf_{zi}$$

然后再套 BM25。

例如：

- title 权重大
- body 权重普通
- anchor 也很重要

这就是工业界常见的 field ranking。

---

# 一句话总结

这章本质是在讲：

> 从 PRP 概率排序原则出发，
> 经历 BIM → IDF → TF saturation → length normalization，
> 最终得到现代最经典排序函数 **BM25 / BM25F**。 :contentReference[oaicite:12]{index=12}