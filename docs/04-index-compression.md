这份文档是斯坦福大学 CS276 课程关于**信息检索（Information Retrieval）**的课件，主要聚焦于**索引压缩（Index Compression）**这一章节。

它详细探讨了在大规模搜索引擎中，如何通过压缩技术减少磁盘空间占用并提升检索速度。以下是该文档的核心内容概览：

### 1. 为什么要进行索引压缩？
文档首先阐述了压缩的必要性，主要基于以下几点：
*   **节省磁盘空间**：处理如 Reuters RCV1 这样的海量数据集时，原始索引体积巨大。
*   **提升速度**：虽然需要解压时间，但现代解压算法很快。减少读取的数据量意味着**从磁盘传输到内存的时间大幅缩短**，整体速度反而提升。
*   **内存容纳**：通过压缩，可以将更多的字典（Dictionary）甚至倒排列表（Postings）保留在内存中，这对于搜索引擎性能至关重要。

### 2. 词汇表（Dictionary）压缩
为了减小字典的体积，文档介绍了以下几种技术：
*   **字符串池（Dictionary-as-a-String）**：将所有词项存储为一个长字符串，并用指针指向每个词的起始位置，避免为每个词项分配固定宽度的空间浪费。
*   **阻塞（Blocking）**：每隔 $k$ 个词项存储一个指针，减少指针总数，虽然查找时需要线性搜索一个小块，但节省了空间。
*   **前缀编码（Front Coding）**：利用排序后的词项通常有共同前缀的特点（如 "automata", "automate"），只存储差异部分。

### 3. 倒排列表（Postings）压缩
这是文档的重点，因为倒排列表通常比字典大得多。核心策略是**间隙编码（Gap Encoding）**：
*   **原理**：不直接存储文档ID（DocID），而是存储排序后DocID之间的**间隙（Gap）**。例如，DocID序列 $33, 47, 154$ 转换为 $33, 14, 107$。由于间隙通常远小于原始DocID，因此更容易压缩。
*   **变长编码技术**：
    *   **Gamma Codes ($\gamma$ codes)**：一种基于二进制长度和偏移量的位级编码，理论上接近最优，但在跨越字节边界时操作较慢。
    *   **变长字节编码（Variable Byte Codes, VB）**：现代系统常用的方法。使用7位存储数据，1位作为继续位（Continuation bit）。它虽然比Gamma编码略占空间，但**字节对齐（Byte-aligned）**，处理速度更快，与计算机内存对齐方式匹配。
    *   **其他编码**：如 Google 曾使用的 Group Variable Integer code 以及 Simple-9/Simple-16 等字对齐编码方案。

### 4. 相关统计定律
文档还引入了两个描述文本集合特性的经验法则，用于解释压缩的潜力：
*   **Heaps' Law**：描述了随着文档集合规模 $T$ 的增长，词汇量 $M$ 的增长趋势（$M = kT^b$）。这帮助预测索引的大小。
*   **Zipf's Law**：描述了词项频率的分布极不均匀，少数词（如 "the"）出现频率极高，而大多数词出现频率极低。这解释了为什么针对小数字（小间隙）优化的变长编码非常有效。

**总结来说**，这份文档系统地讲解了从简单的字符串存储优化到复杂的变长整数编码技术，旨在解决海量数据下索引存储与查询效率的平衡问题。








# 例题 1：Heaps' Law

## 1. Heaps' Law 是什么

Heaps' Law 用来描述：**随着文本集合变大，不同词（vocabulary）的数量也会增加，但增长速度会越来越慢。**

它常用于估计：

- 一个大语料库会有多少个不同词
- 倒排索引中的 dictionary 会有多大
- 索引压缩前大概需要多少空间

它的经验公式是：

$$
M = kT^b
$$

其中：

- $M$：词汇表大小（distinct terms 数量）
- $T$：语料中的 token 总数
- $k,b$：经验参数
- 通常 $b \approx 0.5$

---

## 2. 这道题在干什么

题目给了两个观测点：

- 前 $10{,}000$ 个 token 中有 $3{,}000$ 个不同词
- 前 $1{,}000{,}000$ 个 token 中有 $30{,}000$ 个不同词

然后又告诉你：

- 总网页数是 $2 \times 10^{10}$
- 每页平均有 $200$ 个 token

题目的目标是：

1. 先根据前两个观测点求出 Heaps' Law 里的 $k$ 和 $b$
2. 再估算整个网页集合的 vocabulary size

---

## 3. 解题过程

### Step 1：列出方程

根据 Heaps' Law：

$$
M = kT^b
$$

可得：

$$
3000 = k(10^4)^b
$$

$$
30000 = k(10^6)^b
$$

---

### Step 2：先求 $b$

两式相除：

$$
\frac{30000}{3000} = \frac{k(10^6)^b}{k(10^4)^b}
$$

约去 $k$：

$$
10 = 10^{2b}
$$

所以：

$$
2b = 1
$$

$$
b = 0.5
$$

---

### Step 3：再求 $k$

把 $b=0.5$ 代回：

$$
3000 = k(10^4)^{0.5}
$$

因为：

$$
(10^4)^{0.5} = 10^2 = 100
$$

所以：

$$
3000 = 100k
$$

$$
k = 30
$$

因此得到经验公式：

$$
M = 30T^{0.5}
$$

---

### Step 4：求整个 collection 的 token 总数

网页总数为：

$$
2 \times 10^{10}
$$

每页平均 $200$ 个 token，所以：

$$
T = (2 \times 10^{10}) \times 200
$$

$$
T = 4 \times 10^{12}
$$

---

### Step 5：代入公式估算 vocabulary size

$$
M = 30(4 \times 10^{12})^{0.5}
$$

先算平方根：

$$
(4 \times 10^{12})^{0.5} = 2 \times 10^6
$$

所以：

$$
M = 30 \times 2 \times 10^6
$$

$$
M = 6 \times 10^7
$$

---

## 4. 最终答案

整个网页集合的 vocabulary size 约为：

$$
6 \times 10^7
$$

也就是：

**约 6000 万个不同词。**

---

## 5. 一句话理解这题

这题本质上是在做：

**先用两个样本点拟合 Heaps' Law，再用它预测超大规模语料的词汇表大小。**



# Chapter 5 Index Compression：重点例题精讲

下面继续按照 **「知识点是什么 → 题目在干什么 → 解题过程 → 一句话总结」** 的格式，把 PPT 里的核心题型继续讲下去。内容都来自你上传的 Chapter 5。:contentReference[oaicite:0]{index=0}

---

# 例题 2：Gap Encoding

## 1. Gap Encoding 是什么

在倒排索引里，postings list 存的是一个 term 出现在哪些 docID 中，例如：

$$
[33,47,154,159,202]
$$

直接存 docID 很浪费，因为 docID 本身通常很大。

Gap Encoding 的核心思想是：

> **只存相邻 docID 的差值（gap）**

因为 docID 是递增排序的，相邻差值通常远小于 docID 本身，更容易压缩。:contentReference[oaicite:1]{index=1}

---

## 2. 这道题在干什么

题目给你一个 postings list，让你把它转换成 gap list。

这是所有 postings compression 的第一步。

---

## 3. 解题过程

原始 postings：

$$
33,47,154,159,202
$$

第一个 docID 原样保存：

$$
33
$$

后面都存差值：

$$
47-33=14
$$

$$
154-47=107
$$

$$
159-154=5
$$

$$
202-159=43
$$

---

## 4. 最终答案

gap list 为：

$$
33,14,107,5,43
$$

---

## 5. 一句话理解这题

> **docID → 相邻差值**
>
> 大整数变小整数，为后续 VB / Gamma 压缩做准备。

---

# 例题 3：Gamma Code

## 1. Gamma Code 是什么

Gamma Code 是一种 **按 bit 压缩整数** 的方法，特别适合压缩 gap。

核心思想：

把整数拆成两部分：

1. **长度（length）**
2. **偏移（offset）**

其中：

- length 用 unary 编码
- offset 用去掉最高位后的二进制表示

PPT 第 37 页用 13 做了标准例子。:contentReference[oaicite:2]{index=2}

---

## 2. 这道题在干什么

题目要求：

> 求整数 $13$ 的 Gamma 编码。

---

## 3. 解题过程

### Step 1：转二进制

$$
13=1101_2
$$

---

### Step 2：拆 offset

去掉最高位：

$$
101
$$

所以：

- offset = `101`
- offset 长度 = 3

---

### Step 3：编码 length

长度 3 的 unary 编码：

$$
1110
$$

---

### Step 4：拼接

$$
1110 + 101
$$

得到：

$$
1110101
$$

---

## 4. 最终答案

$$
\boxed{1110101}
$$

---

## 5. 一句话理解这题

> **Gamma = unary(length) + offset**
>
> 很适合压缩小 gap。

---

# 例题 4：Variable Byte (VB) Encoding

## 1. VB Encoding 是什么

VB 是工业界更常用的 postings 压缩方法。:contentReference[oaicite:3]{index=3}

规则：

- 每个 byte 用 **7 bit 存数据**
- 最高位是 continuation bit
- 最后一个 byte 的最高位设为 1

优点：

- 比 gamma 更快
- byte 对齐，CPU 友好

---

## 2. 这道题在干什么

题目要求：

> 对 gap = $5$ 做 VB 编码。

---

## 3. 解题过程

因为：

$$
5 < 128
$$

所以一个 byte 就够。

---

### Step 1：写成 7 bit

$$
5 = 0000101
$$

补足 7 位：

$$
0000101
$$

---

### Step 2：加 continuation bit

因为这是最后一个 byte，最高位设为 1：

$$
10000101
$$

---

## 4. 最终答案

$$
\boxed{10000101}
$$

---

## 5. 一句话理解这题

> **VB 就是每 7 位切一组，最后一组前面补 1。**

---

# 例题 5：Blocking（Dictionary Compression）

## 1. Blocking 是什么

Dictionary-as-a-string 时，每个 term 都存 pointer 很浪费。

Blocking 的做法是：

> **每 $k$ 个词只存一个 pointer**

PPT 用的是：

$$
k=4
$$

:contentReference[oaicite:4]{index=4}

---

## 2. 这道题在干什么

题目要比较：

- 不 blocking
- blocking 后

pointer 空间差多少。

---

## 3. 解题过程

假设：

- 每个 pointer = 3 bytes
- block size = 4

---

### 不 blocking

4 个词都存 pointer：

$$
4 \times 3 = 12 \text{ bytes}
$$

---

### blocking 后

只存 1 个 pointer：

$$
3 \text{ bytes}
$$

另外需要记录 4 个 term length：

$$
4 \text{ bytes}
$$

总计：

$$
3+4=7 \text{ bytes}
$$

---

### 节省空间

$$
12-7=5 \text{ bytes}
$$

---

## 4. 最终答案

每 4 个 term 节省：

$$
\boxed{5\text{ bytes}}
$$

---

## 5. 一句话理解这题

> **用少量 term length 换大量 pointer 空间。**

---

# 高频考点总结

这章最容易考的 5 类题：

1. **Heaps' Law：估算词表**
2. **Gap Encoding：docID 转 gap**
3. **Gamma：bit-level 压缩**
4. **VB：byte-level 压缩**
5. **Blocking：dictionary 压缩**

---

# 推荐刷题顺序

建议按这个顺序掌握：

$$
\text{Heaps} \rightarrow \text{Gap} \rightarrow \text{Gamma} \rightarrow \text{VB} \rightarrow \text{Blocking}
$$

因为 postings compression 的知识是层层递进的。
