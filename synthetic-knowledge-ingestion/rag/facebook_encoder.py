from transformers import AutoModel, AutoTokenizer


class FacebookEncoder:
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")

    @classmethod
    def encode(cls, query):
        inputs = cls.tokenizer(
            query,
            add_special_tokens=True,
            return_tensors="pt",
            truncation="only_first",
            padding="longest",
            return_token_type_ids=False,
        )
        outputs = cls.model(**inputs)
        embeddings = cls.mean_pooling(outputs[0], inputs["attention_mask"])
        return embeddings

    @classmethod
    def mean_pooling(cls, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
