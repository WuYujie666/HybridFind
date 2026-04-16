# Hybrid Document Retrieval with BM25, TF-IDF, and Reciprocal Rank Fusion

Course: [Course Name]  
Team Members: [Name 1], [Name 2], [Name 3]  
Date: [Date]

## Abstract

This project implements a lightweight information retrieval system for document search. The system combines BM25 keyword retrieval and TF-IDF vector similarity, and then merges the two ranking lists with Reciprocal Rank Fusion (RRF). The goal of the project is to build a simple but complete hybrid retrieval pipeline that can index a collection of text documents and return relevant results for user queries. The current implementation is designed as a small-scale prototype and focuses on clarity, interpretability, and practical usability for course-level IR study.

## 1. Introduction

Information retrieval systems are designed to help users find relevant information from a large collection of documents. A common challenge in retrieval is that different ranking methods capture different aspects of relevance. Keyword-based methods are effective when the query shares exact terms with the target documents, but they may fail when relevant documents use different wording. In contrast, vector-based similarity methods can capture broader textual relatedness, but they may not always preserve the precision of exact term matching.

To address this issue, our project explores a hybrid retrieval approach. Instead of relying on a single ranking strategy, we combine lexical matching and vector similarity in one system. This design allows us to benefit from both exact keyword relevance and broader term-distribution similarity. The project is intended as a compact IR prototype that demonstrates the full retrieval workflow, including indexing, query processing, ranking, and result presentation.

## 2. System Design

The system takes a directory of text files as input and treats each file as one document. During indexing, each document is read, normalized, tokenized, and stored together with its identifier and metadata. The same document collection is then processed by two retrieval components.

The first component is BM25, a classic lexical retrieval algorithm widely used in search systems. BM25 scores documents based on term frequency, inverse document frequency, and document length normalization. This component is responsible for strong keyword-based matching.

The second component is a TF-IDF vector search module. Each document is converted into a sparse TF-IDF vector, and the query is represented in the same form. Cosine similarity is then used to measure the similarity between the query vector and each document vector. Although this is not a neural embedding method, it still provides a useful vector-space ranking signal.

After both ranking lists are produced, the system applies Reciprocal Rank Fusion (RRF) to combine them. RRF does not depend on raw score calibration between methods. Instead, it uses ranking positions to generate a final fused order. This makes the hybrid design more stable and easier to tune than directly summing two unrelated scores.

## 3. Implementation

The project is implemented in Python as a lightweight command-line retrieval tool. It supports indexing local text collections and running search queries from the terminal. The implementation includes the following modules:

- document preprocessing and tokenization
- BM25 retrieval
- TF-IDF vector construction and cosine similarity scoring
- weighted rank fusion with RRF
- metadata-aware search filtering
- command-line indexing and search interface

In the current workflow, the system indexes `.txt`, `.md`, and `.rst` files from a given directory. Each file is stored as a document together with basic metadata such as filename and path. At query time, the system runs both BM25 and TF-IDF retrieval over the indexed collection, fuses the results, and returns the top-ranked documents. The output includes the document identifier, ranking score, and a short preview of the document text.

The implementation is intentionally lightweight and does not rely on external search engines or vector databases. This makes the project easy to inspect, explain, and extend in an academic setting.

## 4. Results and Discussion

The current system is able to build an index over a local document collection and return ranked results for user queries. From a project perspective, this demonstrates that the complete retrieval pipeline is functional: documents can be ingested, indexed, searched, and displayed through a simple interface.

One important strength of this project is its interpretability. BM25 provides a clear lexical relevance signal, while TF-IDF cosine similarity offers a second ranking perspective based on vector-space similarity. By combining them with RRF, the system avoids depending entirely on one retrieval strategy. This makes the project a useful baseline for understanding hybrid retrieval in practice.

At the same time, the current version has several limitations. First, the retrieval unit is the full document rather than a smaller passage or chunk, so the output may be less precise for long texts. Second, the vector component is based on TF-IDF rather than neural embeddings, which means its semantic capability is limited compared with modern dense retrieval methods. Third, the current evaluation is mainly demonstration-oriented, and the project has not yet been validated on a large benchmark dataset with formal IR metrics.

Even with these limitations, the project is still meaningful as a course assignment because it presents a complete and understandable IR workflow. It also creates a strong baseline that can be improved later through chunk-based retrieval, embedding-based representations, reranking, or more systematic evaluation.

## 5. Conclusion

This project presents a compact hybrid document retrieval system that combines BM25, TF-IDF vector similarity, and Reciprocal Rank Fusion. The system is able to index text documents, process user queries, and return ranked document-level results through a simple command-line interface. Although the current implementation is intentionally lightweight, it demonstrates the core ideas of hybrid information retrieval in a complete and practical form.

As a course project, this work is valuable because it connects retrieval theory with an actual working prototype. It shows how different ranking signals can be integrated into one pipeline and provides a foundation for future extensions such as finer-grained retrieval, stronger semantic representations, and quantitative evaluation.
