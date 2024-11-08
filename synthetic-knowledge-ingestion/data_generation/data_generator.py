import asyncio
import json
import logging
import os
import pathlib
import sys

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from data_generation.ski_mode import SKIMode
from data_generation.utils import get_ngram_sentences
from llm.llm_connector import LLMConnector
from prompts.question import question_prompt
from prompts.question_answer import question_answer_prompt


class DataGenerator:

    q_prompt = question_prompt  # used for SKI-Q, SKI-QC
    qa_prompt = question_answer_prompt  # used for SKI-QA, SKI-QCA
    data_place_holder = "{passed_in_data}"
    base_path = pathlib.Path(__file__).parent.parent.absolute()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    @classmethod
    async def generate_data(
        cls,
        task: str,
        ngram: int,
        generate_mode: SKIMode = None,
        for_training: bool = False,
        target_count: int = sys.maxsize,
    ):

        data_path = f"{cls.base_path}/datasets/{task}"

        # if data is not there, first download it
        if not os.path.exists(f"{data_path}"):
            cls.logger.info(f"downloading data for {task}")
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
                task
            )
            out_dir = f"{cls.base_path}/datasets"
            data_path = util.download_and_unzip(url, out_dir)
        else:
            cls.logger.info(f"data already downloaded for {task}")

        dataset = "test" if not for_training else "train"
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split=dataset
        )
        full_data = {"corpus": corpus, "queries": queries, "qrels": qrels}
        cls.logger.info(
            f"data loaded: corp length {len(corpus)}, queries length {len(queries)}, qrels length {len(qrels)}"
        )

        rag_questions = set()
        # if there are already clean data, use it
        with open(f"{cls.base_path}/datasets/final_data/{task}/final.json") as f:
            data = json.load(f)
            for item in data:
                rag_questions.add(item.get("id"))

        cls.logger.info(f"rag questions length: {len(rag_questions)}")
        seen_questions = []
        articles = []
        count = 0

        question_set = rag_questions if len(rag_questions) > 0 else qrels

        for qrel in question_set:
            q_articles = qrels[qrel]
            article_ids = q_articles.keys()
            if count < target_count:
                seen_questions.append(qrel)
                count += 1

                for article_id in article_ids:
                    content = corpus[article_id].get("text", "")
                    articles.append({article_id: content})
            else:
                break

        cls.logger.info(f"total articles length: {len(articles)}")
        cls.logger.info(f"sanity check: {articles[0]}")
        expanded_data = []

        if generate_mode:
            groups_of_ten = [articles[i : i + 10] for i in range(0, len(articles), 10)]
            for ind, cur_articles in enumerate(groups_of_ten):
                cls.logger.info(f"checking group {ind}")
                tasks = [
                    cls.generate_question(
                        *(next(iter(x.items()))), ngram, generate_mode
                    )
                    for x in cur_articles
                ]
                results = await asyncio.gather(*tasks)

                for result in results:
                    expanded_data.append(result)

        return expanded_data, articles, seen_questions, full_data

    @classmethod
    async def generate_question(
        cls,
        article_id,
        doc,
        ngram,
        generate_mode: SKIMode,
    ):

        cls.logger.info(f"parsing doc into gram: {ngram}")
        final_list = []

        sentences = get_ngram_sentences(doc, ngram)
        sentences_str = "\n*".join(sentences)

        easy_parse = generate_mode in [SKIMode.Q, SKIMode.QC, SKIMode.QCASM]
        prompt_template = cls.q_prompt if easy_parse else cls.qa_prompt
        final_prompt = prompt_template.replace(
            cls.data_place_holder, f"*{sentences_str}"
        )

        try:
            response = await LLMConnector.get_response(
                [{"role": "user", "content": final_prompt}],
            )
        except Exception as e:
            cls.logger.error(
                f"Error generating question for article {article_id} with error: {e}"
            )
            return {article_id: []}

        if easy_parse:
            question_list = response.split("\n")
            question_list = [i.split(".")[1] if "." in i else i for i in question_list]
            cls.logger.debug(f"cleaned question list is {question_list}")

            cls.logger.debug(sentences)

            if generate_mode == SKIMode.Q:
                final_list = [{"question": question} for question in question_list]

            elif generate_mode == SKIMode.QC:
                final_list = [
                    {"question": question, "context": context}
                    for question, context in zip(question_list, sentences)
                ]
            elif generate_mode == SKIMode.QCASM:
                final_list = [
                    " ".join(
                        [
                            f"{question} {context}"
                            for question, context in zip(question_list, sentences)
                        ]
                    )
                ]
            else:
                raise Exception(f"no easy parse logic for {generate_mode}")
        else:
            try:
                res = json.loads(response)
                question_list = []
                answer_list = []
                for item in res:
                    q = item["q"]
                    answer = item["a"]
                    question_list.append(q)
                    answer_list.append(answer)
            except Exception:
                cls.logger.error(f"failed to parse response: {response}")
                return {article_id: []}

            if generate_mode == SKIMode.QA:
                final_list = [
                    {"question": question, "answer": answer}
                    for question, answer in zip(question_list, answer_list)
                ]
            elif generate_mode == SKIMode.QAASM:
                final_list = [
                    " ".join(
                        [
                            f"{question} {answer}"
                            for question, answer in zip(question_list, sentences)
                        ]
                    )
                ]
            elif generate_mode == SKIMode.QCA:
                final_list = [
                    {"question": question, "context": context, "answer": answer}
                    for question, context, answer in zip(
                        question_list, sentences, answer_list
                    )
                ]
            elif generate_mode == SKIMode.QCAASM:
                final_list = [
                    " ".join(
                        [
                            f"{question} {context} {answer}"
                            for question, context, answer in zip(
                                question_list, sentences, answer_list
                            )
                        ]
                    )
                ]
            else:
                raise Exception(f"no logic for {generate_mode}")

        return {article_id: final_list}


if __name__ == "__main__":
    task = "nq"
    ngram = 1
    ski_mode = SKIMode.QC
    target_count = 10

    asyncio.run(
        DataGenerator.generate_data(
            task, ngram, generate_mode=ski_mode, target_count=target_count
        )
    )
