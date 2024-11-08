import collections
import logging

import numpy as np
from rank_bm25 import BM25Okapi
from scipy import spatial


class Retriever:
    def __init__(self, model=None):
        # when model is NONE, then retriever is using BM25
        self.model = model

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def load_data(self, docs, is_article=False):

        self.database = []
        self.corpus = []

        if not is_article:
            # for non-article, the format is:
            # [ {doc_id: [{"question": "", "context": ""}]} ] for QC
            # or
            # [ {doc_id: [{"question": ""}} ] for Q

            for doc in docs:
                doc_id, qa_pairs = next(iter(doc.items()))
                self.logger.debug(f"preparing {doc_id}")

                for qa_pair in qa_pairs:
                    question = qa_pair.get("question")
                    # if Q, then no context, thus the content should be the question
                    answer = qa_pair.get("context", question)

                    if self.model:
                        # if using embedding mode, prepare by generating embedding
                        embedding = self.model.encode(question)
                        self.database.append(
                            {
                                "id": doc_id,
                                "question": question,
                                "answer": answer,
                                "embedding": embedding.tolist()[0],
                            }
                        )
                    else:
                        # if using BM25, prepare by generating corpus
                        self.database.append(
                            {
                                "id": doc_id,
                                "question": question,
                                "answer": answer,
                            }
                        )
                        self.corpus.append(question.split(" "))
        else:
            # for article the data format is: [{article_id: article_content}]

            for doc in docs:
                doc_id, doc_content = next(iter(doc.items()))
                self.logger.debug(f"preparing {doc_id}")

                if isinstance(doc_content, list):
                    doc_content = doc_content[0]

                if self.model:
                    self.logger.info("using embedding")
                    # if using embedding mode, prepare by generating embedding
                    try:
                        query_embedding = self.model.encode(doc_content).tolist()[0]
                    except:
                        query_embedding = self.model.encode(doc_content)[0]
                    self.database.append(
                        {
                            "id": doc_id,
                            "question": "",
                            "answer": doc_content,
                            "embedding": query_embedding,
                        }
                    )
                else:
                    # if using BM25, prepare by generating corpus
                    self.logger.info("using bm25")
                    self.database.append(
                        {
                            "id": doc_id,
                            "question": "",
                            "answer": doc_content,
                        }
                    )
                    self.corpus.append(doc_content.split(" "))

        if not self.model:
            self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query, top_k=100):
        if self.model:
            # if using embedding mode, retrieve by comparing embedding
            return self.embedding_retrieve(query, top_k)
        else:
            return self.bm25_retrieve(query, top_k)

    def embedding_retrieve(
        self,
        query,
        top_n,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    ):

        self.logger.info(f"retrieve for query embedding: {query}")

        if isinstance(query, dict):
            query = query["text"]

        try:
            query_embedding = self.model.encode(query).tolist()[0]
        except:
            query_embedding = self.model.encode(query)[0]

        results = [
            (
                item["id"],
                item["question"],
                item["answer"],
                relatedness_fn(query_embedding, item["embedding"]),
            )
            for item in self.database
        ]

        self.logger.info("calculation completed")
        results.sort(key=lambda x: x[-1], reverse=True)

        # gather value of the same
        clean_result = collections.defaultdict(int)
        total = len(results)
        self.logger.info(f"total results: {total}")

        # use location of the article to calcualte score
        times = collections.defaultdict(int)
        for ind, result in enumerate(results):
            id = result[0]
            if id not in clean_result:
                if len(clean_result) == top_n:
                    break

                clean_result[id] += total - ind
                times[id] += 1

        # make sure the returned dict is sorted based on its value
        clean_result = dict(
            sorted(clean_result.items(), key=lambda item: item[1], reverse=True)
        )

        # clean_result format: {doc_id: score}
        return clean_result, results[:top_n]

    def bm25_retrieve(self, query, top_n):
        tokenized_query = query.split(" ")
        self.logger.info(f"retrieving with bm 25 for query: {query}")
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[::-1][:top_n]
        results = []

        clean_result = collections.defaultdict(int)
        total = len(scores)
        self.logger.info(f"total results: {total}")

        for ind, index in enumerate(top_k_idx):
            result = self.database[index]
            results.append(result)
            id = result["id"]
            if id not in clean_result:
                if len(clean_result) == top_n:
                    break

                clean_result[id] += total - ind

        clean_result = dict(
            sorted(clean_result.items(), key=lambda item: item[1], reverse=True)
        )
        return clean_result, results[:top_n]
