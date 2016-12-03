import nltk
import numpy as np
import os
import string
import math
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
#print stopwords.words("english")
#raw_input()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import pandas as pd
import re
from bs4 import BeautifulSoup

def word_occurence_in_vocab(total_class_labels,col,train_data_features):
    count =0
    for i in range(total_class_labels):
        if train_data_features[i, col] != 0.0:
            count+=train_data_features[i, col]
    return count

def naive_bayes(train_data_features_copy,total_class_labels,unique_words_in_class,test_word,vocab,feature_names):
    compute_prob = {}
    found_col = -1
    for col in train_data_features.nonzero()[1]:
        if test_word == feature_names[col]:
            found_col = col
            break


    print "found_col",found_col
    raw_input()
    if found_col == -1:
        return -1
    for i in range(total_class_labels):
        #for col in train_data_features.nonzero()[1]:
        if train_data_features[i, found_col] != 0:
            #print "i", i
            #print "col",found_col
            #raw_input()
            compute_prob[i] = (train_data_features[i, found_col] + 1.0)/(unique_words_in_class[i]+1)#P(W/C_i)
            #print compute_prob[i]
            #raw_input()

            P_C = (1.0/total_class_labels);
            #print "total_class_labels" , P_C
            P_W = word_occurence_in_vocab(total_class_labels,col,train_data_features)
            #print "P_W",P_W #
            compute_prob[i] = np.log(compute_prob[i]) + math.log(P_C) - math.log(P_W)

        # now i have dictionary with all probabilities
    get_max = [(value,key) for key,value in compute_prob.items()]
    return max(get_max)[1]




def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

y_train =[]
path = 'files'
token_dict = []
label = []
#files_fixed = ['ai.txt','alice.txt','astrology.txt']
#
for subdir, dirs, files in os.walk(path):
    #print files
    #raw_input()

    for file in files:

        #print file
        file_path = subdir + os.path.sep + file
        shakes = open(file_path, 'r')
        text = shakes.readlines()
        for i in range(len(text)):
            lowers = review_to_words(text[i])
             #no_punctuation = lowers.translate(None, string.punctuation)
            token_dict.append(lowers) #corpus list of raw sentences
            y_train.append(file)
            #print token_dict
            #raw_input()

vectorizer = TfidfVectorizer(min_df = 1)
x_train = vectorizer.fit_transform(token_dict)

#training svm
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size = .2,random_state = 42)
svm = SVC(C = 1000000,gamma = 'auto',kernel ='rbf')
svm.fit(x_train,y_train)

clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(200,),random_state=1)
clf.fit(x_train,y_train)

pred = svm.predict(x_test)
pred_nn=clf.predict(x_test)

print(svm.score(x_test,y_test))
print(clf.score(x_test,y_test))
'''
token_dict_test = []
path1 ='/home/rahul/Documents/Learn_ML/ML_project/test'
y_test = []

for subdir1, dirs1, files_test in os.walk(path1):
    #print files
    #raw_input()

    for file_test in files_test:
        y_test.append(file_test)
        #print file
        file_path1 = subdir1 + os.path.sep + file_test
        shakes_test = open(file_path1, 'r')
        text_test = shakes_test.read()
        lowers_test = review_to_words(text_test)
        #no_punctuation = lowers.translate(None, string.punctuation)
        token_dict_test.append(lowers_test) #corpus list of raw sentences
        #print token_dict
        #raw_input()

vectorizer_test = TfidfVectorizer(min_df = 1)
x_test = vectorizer_test.fit_transform(token_dict)
x_test = x_test.toarray()
'''


'''svm = SVC(C = 1000000,gamma = 'auto',kernel ='rbf')
svm.fit(x_train,y_train)

pred = svm.predict(x_test)

print(svm.score(x_test,y_test))'''
