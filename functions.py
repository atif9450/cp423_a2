import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from math import log10

def process_file(file):
    text = file.read().lower() #read text and convert to lowercase
    
    #tokenize and load stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    stop_words = stopwords.words('english')
    
    #construct new word set without stopwords
    new_tokenized = []
    for token in tokenized:
        if token not in stop_words:
            new_tokenized.append(token)
    
    return new_tokenized

def construct_positional_index(path_to_files):
    doc_indices = {} #for enumerating the documents
    positional_index = {} #this will be our positional index
    index = 0 #to keep track of document enumeration
    files = os.listdir(path_to_files) #list of files to walk through
    for f in files: #loop through files
        doc_indices.update({f: index}) #update dict for document enumeration
        with open(path_to_files + f, 'r') as file: #open file
            words = process_file(file) #process file
            for word in words: #update positional index for each word
                
                #if word is not already in vocabulary, initialize values in positional index
                keys = list(positional_index.keys())
                if word not in keys:
                    array = np.zeros(len(files))
                    positional_index.update({word: [0, array]})

                positional_index[word][1][index] += 1 #update term frequency for corresponding document

            words = list(set(words))
            for word in words:
                positional_index[word][0] += 1 #update document frequency for each term
        index += 1 #update index for document enumeration
    
    return doc_indices, positional_index

def construct_binary_tfidf(doc_indices, positional_index):
    words = list(positional_index.keys()) #get vocabulary
    docs = list(doc_indices.keys()) #get document names
    matrix = np.zeros((len(words), len(docs))) #initialize TF-IDF matrix

    word_index = 0 #keep track of word index
    for word in words: #iterate through vocabulary
        doc_freq = positional_index[word][0] #get document frequency of term
        idf = log10((len(docs) + 1) / doc_freq) #calculate IDF for term
        array = positional_index[word][1] #get term frequency array for term
        vector = np.array([1 if x!=0 else 0 for x in array]) #construct binary vector
        vector = vector * idf #calculate TF-IDF vector
        matrix[word_index] = vector #insert vector into matrix
        word_index += 1 #update word index

    return matrix

def construct_raw_count_tfidf(doc_indices, positional_index):
    words = list(positional_index.keys()) #get vocabulary
    docs = list(doc_indices.keys()) #get document names
    matrix = np.zeros((len(words), len(docs))) #initialize TF-IDF matrix

    word_index = 0 #keep track of word index
    for word in words: #iterate through vocabulary
        doc_freq = positional_index[word][0] #get document frequency of term
        idf = log10((len(docs) + 1) / doc_freq) #calculate IDF for term
        array = positional_index[word][1] #get term frequency array for term
        vector = array * idf #calculate TF-IDF vector
        matrix[word_index] = vector #insert vector into matrix
        word_index += 1 #update word index

    return matrix

def construct_term_freq_tfidf(doc_indices, positional_index):
    words = list(positional_index.keys()) #get vocabulary
    docs = list(doc_indices.keys()) #get document names
    matrix = np.zeros((len(words), len(docs))) #initialize TF-IDF matrix

    #calculate sum of term frequencies
    raw_matrix = construct_raw_count_tfidf(doc_indices, positional_index) #get raw matrix
    sum_vector = raw_matrix.sum(axis=0) #get term-frequency sum for each document
    raw_matrix = None
    
    word_index = 0 #keep track of word index
    for word in words: #iterate through vocabulary
        doc_freq = positional_index[word][0] #get document frequency of term
        idf = log10((len(docs) + 1) / doc_freq) #calculate IDF for term
        array = positional_index[word][1] #get term frequency array for term
        vector = array / sum_vector #calculate term frequency
        vector = vector * idf #calculate TF-IDF vector
        matrix[word_index] = vector #insert vector into matrix
        word_index += 1 #update word index

    return matrix

def construct_log_norm_tfidf(doc_indices, positional_index):
    words = list(positional_index.keys()) #get vocabulary
    docs = list(doc_indices.keys()) #get document names
    matrix = np.zeros((len(words), len(docs))) #initialize TF-IDF matrix

    word_index = 0 #keep track of word index
    for word in words: #iterate through vocabulary
        doc_freq = positional_index[word][0] #get document frequency of term
        idf = log10((len(docs) + 1) / doc_freq) #calculate IDF for term
        array = positional_index[word][1] #get term frequency array for term
        vector = array + 1
        vector = np.log10(vector)
        vector = vector * idf #calculate TF-IDF vector
        matrix[word_index] = vector #insert vector into matrix
        word_index += 1 #update word index

    return matrix

def construct_double_norm_tfidf(doc_indices, positional_index):
    words = list(positional_index.keys()) #get vocabulary
    docs = list(doc_indices.keys()) #get document names
    matrix = np.zeros((len(words), len(docs))) #initialize TF-IDF matrix

    raw_matrix = construct_raw_count_tfidf(doc_indices, positional_index) #get raw matrix
    max_vector = raw_matrix.max(axis=0) #get max term-frequency for each document
    raw_matrix = None

    word_index = 0 #keep track of word index
    for word in words: #iterate through vocabulary
        doc_freq = positional_index[word][0] #get document frequency of term
        idf = log10((len(docs) + 1) / doc_freq) #calculate IDF for term
        array = positional_index[word][1] #get term frequency array for term
        vector = 0.5 + 0.5 * (array / max_vector)
        vector = vector * idf #calculate TF-IDF vector
        matrix[word_index] = vector #insert vector into matrix
        word_index += 1 #update word index

    return matrix

def cosine_sim(x, y):
    a = np.dot(x, y)
    b = np.linalg.norm(x)
    c = np.linalg.norm(y)
    sim = a / (b * c)
    return sim

def get_top_5(query, doc_indices, matrix):
    sims = {} #keep track of document-query similarities
    docs = list(doc_indices.keys()) #get list of documents
    for i in range(len(docs)): #iterate through number of documents
        doc_vector = matrix[:, i] #get column vector representing document
        sim = cosine_sim(query, doc_vector.T) #calculate cosine similarity of query and document vectors
        sims.update({i: sim}) #update similarities dictionary

    sims = dict(sorted(sims.items(), key=lambda item: item[1])) #sort similarities-dictionary on values
    indices = list(sims.keys()) #get document indices from sorted dictionary
    indices = indices[:5] #only keep first 5 document indices

    top_5 = [] #collect document names to return
    doc_names = list(doc_indices.keys()) #get document names
    for name in doc_names: #iterate through all documents
        if doc_indices[name] in indices: #if document is among top 5, append to top_5
            top_5.append(name)

    return top_5

def construct_queries(query, positional_index):
    words = list(positional_index) #get vocabulary
    binary_vector = np.array([1 if x in query else 0 for x in words]) #construct bit-vector for query

    #calculate raw-count query vector
    vector = np.zeros(len(words))
    for q in query:
        try:
            index = words.index(q)
            vector[index] += 1
        except:
            pass

    term_freq_vector = vector / vector.sum() #perform term-frequency transformation on query
    log_norm_vector = np.log10(1 + vector) #perform log normalization on query
    double_norm_vector = 0.5 + 0.5 * (vector / vector.max()) #perform double normalization on query
    return binary_vector, vector, term_freq_vector, log_norm_vector, double_norm_vector