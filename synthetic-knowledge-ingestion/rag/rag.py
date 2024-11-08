import asyncio
import json
import logging

from facebook_encoder import FacebookEncoder
from retriever import Retriever

from data_generation.data_generator import DataGenerator
from data_generation.ski_mode import SKIMode
from evaluation.evaluator import Evaluator
from llm.llm_connector import LLMConnector
from prompts.rag import rag_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

data_place_holder = "{passed_in_data}"

async def evaluate_retrieval(
    task, ngram, target_count, retriever: Retriever, ski_mode: SKIMode = None
):

    qrels, final_results, _ = await retrieve(
        task, ngram, target_count, retriever, ski_mode
    )

    performance = Evaluator.evaluate_retrieval(final_results, qrels)
    logger.info(f"final retrieval results are: {performance}")


async def retrieve(
    task, ngram, target_count, retriever: Retriever, ski_mode: SKIMode = None
):

    if not ski_mode:
        logger.info(f"Original Aritcle for task: {task}; ngram: {ngram}")

    else:
        assert ski_mode in [SKIMode.Q, SKIMode.QC, SKIMode.QCASM]
        logger.info(f"SKI mode: {ski_mode} for task: {task}; ngram: {ngram}")

    (
        augmented_data,
        articles,
        seen_questions,
        full_data,
    ) = await DataGenerator.generate_data(
        task, ngram, ski_mode, target_count=target_count
    )

    if not ski_mode:
        retriever.load_data(articles, is_article=True)
    elif ski_mode == SKIMode.QCASM:
        retriever.load_data(augmented_data, is_article=True)
    else:
        retriever.load_data(augmented_data)

    logger.info("data loaded")
    logger.info(f"starting quering")

    retrieval_count = max(Evaluator.k_values)
    final_results = {}
    all_retrieved_articles = {}
    queries, qrels = full_data.get("queries"), full_data.get("qrels")

    # read data and perform retrieval
    for query_id in seen_questions:
        logger.debug(f"checking query {query_id}")
        query = queries.get(query_id)

        times = 0
        while times < 10:
            try:
                result_ids, results = retriever.retrieve(query, retrieval_count)
                final_results[query_id] = result_ids

                all_retrieved_articles[query_id] = {"articles": results, "query": query}

                logger.debug(f"result is: {result_ids}")
                logger.debug(f"qrel is: {qrels[query_id]}")

                break
            except Exception as e:
                logger.error(f"error: {e}")
                times += 1
    return qrels, final_results, all_retrieved_articles


async def evaluate_rag(
    task, ngram, target_count, retriever: Retriever, ski_mode: SKIMode = None
):  
    _, _, all_retrieved_articles = await retrieve(
        task, ngram, target_count, retriever, ski_mode
    )

    answer_path = f"./datasets/final_data/{task}/final.json"
    ground_truth = {}
    with open(answer_path, "r") as f:
        answer = json.load(f)
        for item in answer:
            id = item["id"]
            answer = item["answer"]
            ground_truth[id] = answer

    res = {}
    test_data = []

    with open("all_retrieved.json", "w") as f:
        json.dump(all_retrieved_articles, f)

    for id in all_retrieved_articles:
        data = all_retrieved_articles[id]
        articles = data.get("articles")
        query = data.get("query")

        test_data.append([id, articles, query])

    groups_of_ten = [test_data[i : i + 10] for i in range(0, len(test_data), 10)]

    for ind, cur_articles in enumerate(groups_of_ten):
        logger.info(f"checking group {ind}")
        tasks = [get_rag_answer(data) for data in cur_articles]
        results = await asyncio.gather(*tasks)

        for result in results:
            res.update(result)

    score = Evaluator.evaluate_f1(res, ground_truth)
    logger.info(f"final rag results are: {score}")


async def get_rag_answer(data, doc_limit=5):
    id, docs, query = data

    articles = ""
    for res in docs[:doc_limit]:
        content = res[2]

        articles += f"* {content} \n\n"
    articles += f" ## Question ## \n {query} \n ## Answer ##: answer within 3 words.\n"
    final_prompt = rag_prompt.replace(data_place_holder, articles)
    response = await LLMConnector.get_response(
        [{"role": "user", "content": final_prompt}]
    )
    return {id: response}


if __name__ == "__main__":
    # use BM 25
    retriever = Retriever()

    # use contriver
    model = FacebookEncoder()
    retriever = Retriever(model)

    task = "nq"
    ngram = 1
    ski_mode = SKIMode.QC
    target_count = 10

    # asyncio.run(evaluate_retrieval(task, ngram, target_count, retriever, ski_mode))
    asyncio.run(evaluate_rag(task, ngram, target_count, retriever, ski_mode))
