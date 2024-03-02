from tf_idf import construct_positional_index, construct_binary_tfidf

docs, pos_index = construct_positional_index('data/')
matrix = construct_binary_tfidf(docs, pos_index)
print(matrix)