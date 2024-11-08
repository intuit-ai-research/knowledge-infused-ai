import nltk

nltk.download("popular")
nltk.download("punkt_tab")


def get_ngram_sentences(paragraph, n_gram=1):
    assert n_gram > 0
    sentences = nltk.sent_tokenize(paragraph)

    n_gram_sentences = []
    for i in range(0, len(sentences), n_gram):
        n_gram_sentences.append(" ".join(sentences[i : i + n_gram]))

    return n_gram_sentences
