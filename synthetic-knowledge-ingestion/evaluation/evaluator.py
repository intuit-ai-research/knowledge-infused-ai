import logging
import re
import string
from collections import Counter

import pytrec_eval


class Evaluator:
    k_values = [1, 3, 5, 10]

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    @classmethod
    def evaluate_retrieval(cls, results, qrels):
        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in cls.k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in cls.k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in cls.k_values])
        recall_string = "recall." + ",".join([str(k) for k in cls.k_values])
        precision_string = "P." + ",".join([str(k) for k in cls.k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in cls.k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in cls.k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            cls.logger.info("\n")
            for k in eval.keys():
                cls.logger.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision

    @classmethod
    def evaluate_f1(cls, results, ground_truths):
        pred = []
        truth = []
        for id in results:
            if id in ground_truths:
                pred.append(results[id])
                truth.append(ground_truths[id])

        scores = 0
        for p, t in zip(pred, truth):
            score = cls.qa_f1_score(p, t)
            scores += score

        return scores / len(pred)

    @classmethod
    def normalize_answer(cls, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def qa_f1_score(cls, prediction, ground_truth):
        if isinstance(ground_truth, list):
            clean = ""
            for g in ground_truth:
                if isinstance(g, list):
                    for gn in g:
                        clean += gn + " "
                else:
                    clean += g + " "
            ground_truth = clean
        normalized_prediction = cls.normalize_answer(prediction)
        normalized_ground_truth = cls.normalize_answer(ground_truth)
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        return cls.f1_score(prediction_tokens, ground_truth_tokens)

    @classmethod
    def f1_score(cls, prediction, ground_truth):
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
