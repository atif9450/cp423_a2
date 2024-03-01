import numpy as np

def construct_tfidf_1(positional_index: dict, docs: list):
    words = list(positional_index.keys())
    num_words = len(words)
    num_docs = len(docs)
    matrix = np.zeros((num_words, num_docs))
    for w in words:
        values = positional_index[w]
        pass