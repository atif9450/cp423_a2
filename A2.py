#all the imports and downloads
#NOT APART OF THE MAIN
#IMPORTANT (PLEASE READ): here the positions of the documents words start at 0
import os
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import string


def fill(out, inp):
    for lines in inp:
        out.write(lines)
    return

#declaring all empty lists and counters
dict = {}
file_num = 0
set_of_words = set()
total_list = []
file_names = []
n_doc_r = 0


#input = ["adveniure", "of", "the", "three"]
#pqury = "true"
#takes input 
pqury = input("Is your query a phrase query? (Enter true or false): ")
input_taken = input("Enter your query like --> Mr. Sherlock Holmes: ")

#lowers input
input = word_tokenize(input_taken.lower()) 

i = 0
j=0



#path to folder with all the files
folder = "C:/Users/1malh/Desktop/cp423/data"

#walking through every single file in the folder
for root, dirs, files in os.walk(folder):

    #getting each file from folder 
    for name in files:
        docs_list_of_words = []
        f = open(name, "r+" , errors="ignore")
        f2 = open("text.txt","w")

        #lowers the files words and places it on another file
        for line in f:
            line = line.lower()
            f2.write(line)
        

        f.close()
        f2.close()

        f = open(name, "w",  errors="ignore")
        f2 = open("text.txt", "r")

        #replaces the original files uppercase words and the rest of words with lowercase from the other file
        fill(f,f2)
        f.close()
        f2.close()

        list = []
        f = open(name, "r+" )
        
        #tokenizing the words from the file
        for line in f:
           

            #i = 0
            str = ''
            fs = word_tokenize(line) 
            #checks if the user inputed a phrase query
            if (pqury.lower() == "true"):
                
                k = []
                for w in fs:
                    #checks if the users phrase query tokenized is equal to the word in the file
                    if w.lower() == input[i].lower():
                        #adding the phrase query words to the string called str so later can filter them
                        str += w
                        #str+=' '
                        
                        if i == (len(input)-1):
                            #if the full phrase query has been added into str add it into list k
                            k.append(str)
                           
                        if i < (len(input)-1):
                            #will use this later to add a space
                            i+=1
                            str+='0000000000'

                        
                    else:
                        i = 0
                        #list.append(w)
                        #appends the rest of the words other than the phrase query to the list to filer their positions later 
                        #to get the phrase query positions comment out the line below
                        ###############################k.append(w)
                j +=1
                list.append(k)
            else:
                #adding tokenized words to list
                list.append(fs)
        
        f.close()
        
        
        #taking out each word from the tokenized list
        for word in list:
           
           #taking each letter from the tokenized word
            for letter in word:

                #only takes the letters that are in the alphabet or numbers
                only_alphnum = re.sub("[^A-Za-z0-9]","",letter)  

                #eliminates '' key
                if (len(only_alphnum) >= 1):
                    if (only_alphnum in stopwords.words('english')):
                        f = 0
                    #only takes the words that are not stopwords
                    else:
                        #if (letter not in string.punctuation):
                        only_alphnum2 = only_alphnum.replace('0000000000', ' ')
                        docs_list_of_words.append(only_alphnum2)

        #docs_list_of_words is added to total_list which is a total list of all words in all docsss
        total_list.extend(docs_list_of_words)
        
        
        #creating a set of words using total_list
        set_of_words.update(total_list)

        
        

        #gets words from the set of all words
        for dict_word_key in set_of_words:
            
            #creates a list called doc and positions that is a value list for the key which in this case is a word
            docs = []
            positions = []
            inds = {}

            #checks if the key (word) is in the dict 
            if dict_word_key not in dict:

                #creates the key value pair to aviod future error
                dict[dict_word_key] = [-1]
            
            
            
            #checks if the dictionary (dict) word is in the file
            if dict_word_key in docs_list_of_words:
                
                

                if file_num not in inds:

                    #creates the key value pair to aviod future error
                    inds[file_num] = [-1]
                word_ind = filter(lambda i: docs_list_of_words[i]==dict_word_key, range(len(docs_list_of_words)))

                for idx in word_ind:
                    
                    positions.append(idx)
                    
                #removing the -1 if it was in docs list as it was originally put there to prevent an error
                if -1 in inds[file_num] :
                    inds[file_num].pop(0)
                
                

                if -1 in dict[dict_word_key]:
                    dict[dict_word_key].pop(0)

                #adds the number of docs retrieved
                n_doc_r += 1
                #adds the docs list as teh vale in the key value pair
                
                inds[file_num] = positions
                
                dict[dict_word_key].append(inds)
                

            
        
        #creates a list of all the files names and the indices are the doc numbers/ids
        file_names.append(name)
        
        #ensures the doc numbers/indices are the same in file_names
        file_num +=1

        
print(dict)
print(n_doc_r)
print(file_names)


