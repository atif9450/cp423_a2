from functions import *
import pickle

docs = None
pos_index = None
binary_matrix = None
raw_count_matrix = None
term_freq_matrix = None
log_norm_matrix = None
double_norm_matrix = None

#try to load pre-built files/matrices; construct if failed
try:
    docs = pickle.load(open('pkl/docs.pkl', 'rb'))
    pos_index = pickle.load(open('pkl/pos_index.pkl', 'rb'))
    binary_matrix = pickle.load(open('pkl/binary.pkl', 'rb'))
    raw_count_matrix = pickle.load(open('pkl/raw_count.pkl', 'rb'))
    term_freq_matrix = pickle.load(open('pkl/term_freq.pkl', 'rb'))
    log_norm_matrix = pickle.load(open('pkl/log_norm.pkl', 'rb'))
    double_norm_matrix = pickle.load(open('pkl/double_norm.pkl', 'rb'))
except:
    docs, pos_index = construct_positional_index('data/')
    binary_matrix = construct_binary_tfidf(docs, pos_index)
    raw_count_matrix = construct_raw_count_tfidf(docs, pos_index)
    term_freq_matrix = construct_term_freq_tfidf(docs, pos_index)
    log_norm_matrix = construct_log_norm_tfidf(docs, pos_index)
    double_norm_matrix = construct_double_norm_tfidf(docs, pos_index)

    pickle.dump(docs, open('pkl/docs.pkl', 'wb'))
    pickle.dump(pos_index, open('pkl/pos_index.pkl', 'wb'))
    pickle.dump(binary_matrix, open('pkl/binary.pkl', 'wb'))
    pickle.dump(raw_count_matrix, open('pkl/raw_count.pkl', 'wb'))
    pickle.dump(term_freq_matrix, open('pkl/term_freq.pkl', 'wb'))
    pickle.dump(log_norm_matrix, open('pkl/log_norm.pkl', 'wb'))
    pickle.dump(double_norm_matrix, open('pkl/double_norm.pkl', 'wb'))

query = input("Please input your query: ") #get user input query
print('\n')
with open('query.txt', 'w+') as f: #process query
    f.write(query)
    
with open('query.txt', 'r') as f:
    query = process_file(f)

#get query vector with different transformations
binary_vector, raw_count_vector, term_freq_vector, log_norm_vector, double_norm_vector = construct_queries(query, pos_index)

#binary
top_5 = get_top_5(binary_vector, docs, binary_matrix)
print("Using binary TF-IDF:")
print(top_5)
print('\n')

#raw count
top_5 = get_top_5(raw_count_vector, docs, raw_count_matrix)
print("Using raw count TF-IDF:")
print(top_5)
print('\n')

#term frequency
top_5 = get_top_5(term_freq_vector, docs, term_freq_matrix)
print("Using term frequency TF-IDF:")
print(top_5)
print('\n')

#log norm
top_5 = get_top_5(log_norm_vector, docs, log_norm_matrix)
print("Using log-normalization TF-IDF:")
print(top_5)
print('\n')

#double norm
top_5 = get_top_5(double_norm_vector, docs, double_norm_matrix)
print("Using double-normalization TF-IDF:")
print(top_5)
print('\n')