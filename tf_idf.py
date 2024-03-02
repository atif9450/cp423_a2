import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from math import log10
from copy import deepcopy

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

                positional_index[word][0] += 1 #update document frequency of term
                positional_index[word][1][index] += 1 #update term frequency for corresponding document
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
