from tf_idf import *

docs, pos_index = construct_positional_index('data/')

matrix1 = construct_binary_tfidf(docs, pos_index)
matrix2 = construct_raw_count_tfidf(docs, pos_index)