# Knowledge Infused AI

This is a research initiative by Intuit AI Research and focuses on techniques and methodologies to incorporate knowledge of various forms to improve the performance of AI systems. Some of our work in this space include -
- Synthetic Knowledge Ingestion :- [Ski](#ski-towards-knowledge-refinement-and-injection-for-enhancing-large-language-models) Knowledge Refinement and Injection for Enhancing Large Language Models
- RAG Context Ranking :- [HyQE](#hyqe-ranking-contexts-with-hypothetical-query-embeddings) Ranking Contexts with Hypothetical Query Embeddings

## :fire: News
- [2024.11] Blog [Enhancing LLMs with Synthetic Knowledge Ingestion: A Novel Approach from Intuit AI Research at EMNLP 2024](https://medium.com/intuit-engineering/enhancing-llms-with-synthetic-knowledge-ingestion-a-novel-approach-from-intuit-ai-research-at-01e8f02b9c46)
- [2024.10] HyQE paper accepted at EMNLP 2024
- [2024.10] Ski paper accepted at EMNLP 2024

## Synthetic Knowledge Ingestion
### Ski: Towards Knowledge Refinement and Injection for Enhancing Large Language Models
[[Paper]](https://arxiv.org/pdf/2410.09629) [[Code]](https://github.com/intuit-ai-research/knowledge-infused-ai/tree/main/synthetic-knowledge-ingestion)

Large language models (LLMs) are proficient in capturing factual knowledge across various domains. However, refining their capabilities on previously seen knowledge or integrating new knowledge from external sources remains a significant challenge. In this work, we propose a novel synthetic knowledge ingestion method called Ski, which leverages fine-grained synthesis, interleaved generation, and assemble augmentation strategies to construct high-quality data representations from raw knowledge sources. We then integrate Ski and its variations with three knowledge injection techniques: Retrieval Augmented Generation (RAG), Supervised Fine-tuning (SFT), and Continual Pre-training (CPT) to inject and refine knowledge in language models. Extensive empirical experiments are conducted on various question-answering tasks spanning finance, biomedicine, and open-generation domains to demonstrate that Ski significantly outperforms baseline methods by facilitating effective knowledge injection. We believe that our work is an important step towards enhancing the factual accuracy of LLM outputs by refining knowledge representation and injection capabilities.


## RAG Context Ranking
### HyQE: Ranking Contexts with Hypothetical Query Embeddings

[[Paper]](https://arxiv.org/abs/2410.15262) [[Code]](https://github.com/zwc662/hyqe)

In retrieval-augmented systems, context ranking techniques are commonly employed to reorder the retrieved contexts based on their relevance to a user query. A standard approach is to measure this relevance through the similarity between contexts and queries in the embedding space. However, such similarity often fails to capture the relevance. Alternatively, large language models (LLMs) have been used for ranking contexts. However, they can encounter scalability issues when the number of candidate contexts grows and the context window sizes of the LLMs remain constrained. Additionally, these approaches require fine-tuning LLMs with domain-specific data. In this work, we introduce a scalable ranking framework that combines embedding similarity and LLM capabilities without requiring LLM fine-tuning. Our framework uses a pre-trained LLM to hypothesize the user query based on the retrieved contexts and ranks the context based on the similarity between the hypothesized queries and the user query. Our framework is efficient at inference time and is compatible with many other retrieval and ranking techniques. Experimental results show that our method improves the ranking performance across multiple benchmarks.
This work has been done in collaboration with Boston University.
