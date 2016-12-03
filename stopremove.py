# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys
# nltk 
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

path = '/home/sachin77/Documents/Sachin/word2vec-sentiment/ML_Project/'
pathRes = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/' 


for file in os.listdir(path):
    if file.endswith('.txt'):
        file_path = path + file
        print(file_path)
        fw = open(pathRes+file,'w');
        shakes = open(file_path, 'r')
        text = shakes.readlines()
        for i in range(len(text)):
            lowers = review_to_words(text[i])
             #no_punctuation = lowers.translate(None, string.punctuation)
            fw.write("{}\n".format(lowers))
            # token_dict.append(lowers) #corpus list of raw sentences
            # y_train.append(file)
        fw.close()