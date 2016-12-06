import nltk
import numpy as np
import os
import string
import math
import sys
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
#print stopwords.words("english")
#raw_input()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import os
# random shuffle
import random
from random import shuffle

# numpy
#import numpy

# classifier
from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import logging
import sys
import scipy.io



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

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.split()

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


number = sys.argv[1]
y_train =[]
path = 'files_'+number
token_dict = []
label = []
svm_accuracy = []
lr_accuracy = []
nn_accuracy = []
Labels = []
k_fold_nn=[]
k_fold_svm = []
k_fold_lr = []
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
            Labels.append(file)
            #print token_dict
            #raw_input()

y = np.array(Labels)
#print y.shape
#raw_input()
vectorizer = TfidfVectorizer(min_df = 1)#,ngram_range = (2,2))
sparse_matrix = vectorizer.fit_transform(token_dict)
#Label_array = np.asarray(Labels)
#Label_array = np.array(Labels).reshape(len(Labels),1);
kf = KFold(n_splits=10,shuffle = True)
for train_index, test_index in kf.split(sparse_matrix):#splitting sparse matrix for 2 fold testing
    x_train_nn = sparse_matrix[train_index]
    x_test_nn = sparse_matrix[test_index]
    #print x_train_nn.shape
    #raw_input()
    #print x_test_nn.shape
    #raw_input()
    #y_train_nn = [i for i in train_index.tolist()]
    #y_test_nn = [i for i in test_index.tolist()]
    y_train_nn = y[train_index]
    y_test_nn =  y[test_index]
    #print y_train_nn.shape
    #print y_test_nn.shape
    #raw_input()
    #print y_train_nn
    #print y_test_nn
    #y_train_nn = Label_array[train_index,0]
    #y_test_nn =  Label_array[test_index,0]
    #y_train_nn = Labels[i for i in train_index]
    #y_test_nn =  Labels[i for i in test_index]
#x_train_nn,x_test_nn,y_train_nn,y_test_nn = train_test_split(sparse_matrix,Labels,test_size = .2,random_state = 42)


    ## NN Classifier
    clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(200,),random_state=1)
    clf.fit(x_train_nn,y_train_nn)
#validationScore_nn.append(clf.score(x_test, y_test))
#nn_accuracy.append(clf.score(x_test_nn, y_test_nn))
    #predict_nn = clf.predict(x_test_nn)
    #print "NN",clf.score(x_test_nn, y_test_nn)

#print accuracy
    #accuracy_nn = open('accuracy_nn_'+number+'.txt','w')

    #accuracy_nn.write("Accuracy of NN with TFIDF is ")
    #accuracy_nn.write(str(clf.score(x_test_nn,y_test_nn)))
    #accuracy_nn.write('\n')
    #accuracy_nn.write('Predicted - Actual')
    #accuracy_nn.write('\n')
    #accuracy_nn.write('\n')
    #for i in range(len(y_test_nn)):
        #accuracy_nn.write(str(predict_nn[i]))
        #accuracy_nn.write('-')
        #accuracy_nn.write(str(y_test_nn[i]))
        #accuracy_nn.write('\n')

    k_fold_nn.append(str(clf.score(x_test_nn,y_test_nn)))


##SVM classifier
#x_train_svm,x_test_svm,y_train_svm,y_test_svm = train_test_split(sparse_matrix,Labels,test_size = .2,random_state = 42)



    svm = SVC(C = 1000000,gamma = 'auto',kernel ='rbf')
    svm.fit(x_train_nn,y_train_nn)
#pred = svm.predict(x_test)
#svm_accuracy.append(svm.score(x_test_svm,y_test_svm))
    #predict_svm = svm.predict(x_test_nn)
# Print accuracy

    #accuracy_svm = open('10-fold_accuracy_svm_'+number+'.txt','w')

    #accuracy_svm.write("Accuracy of SVM with TFIDF is ")
    #accuracy_svm.write(str(svm.score(x_test_nn,y_test_nn)))
    k_fold_svm.append(str(svm.score(x_test_nn,y_test_nn)))
#LR###################################################################
#x_train_lr,x_test_lr,y_train_lr,y_test_lr = train_test_split(sparse_matrix,Labels,test_size = .2,random_state = 42)

    classifier  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001,multi_class = 'ovr')

    classifier.fit(x_train_nn, y_train_nn)
    k_fold_lr.append(str(classifier.score(x_test_nn,y_test_nn)))
#Print accuracy
#predict_lr = classifier.predict(x_test_lr)
##############################################

#accuracy_lr = open('accuracy_lr_'+number+'.txt','w')

#accuracy_lr.write("Accuracy of LR with TFIDF is ")
#accuracy_lr.write(str(classifier.score(x_test_lr,y_test_lr)))
#accuracy_lr.write('\n')
#accuracy_lr.write('Predicted - Actual')
#accuracy_lr.write('\n')
#accuracy_lr.write('\n')
#for i in range(len(y_test_lr)):
#    accuracy_lr.write(predict_lr[i])
#    accuracy_lr.write('-')
#    accuracy_lr.write(y_test_lr[i])
#    accuracy_lr.write('\n')
sum = 0
k_fold_results = open('10_fold_classifiers'+number+'.txt','w')
k_fold_results.write('K_fold_nn')
k_fold_results.write('\n')
for i in range(10):
    k_fold_results.write(k_fold_nn[i])
    sum = sum+float(k_fold_nn[i])
    k_fold_results.write('\n')
sum = sum/10
k_fold_results.write('sum')
k_fold_results.write('\n')
k_fold_results.write(str(sum))
k_fold_results.write('\n')
sum = 0

k_fold_results.write('K_fold_svm')
k_fold_results.write('\n')
for i in range(10):
    k_fold_results.write(k_fold_svm[i])
    sum = sum+float(k_fold_svm[i])
    k_fold_results.write('\n')
sum = sum/10
k_fold_results.write('sum')
k_fold_results.write('\n')
k_fold_results.write(str(sum))
k_fold_results.write('\n')
sum = 0

k_fold_results.write('K_fold_lr')
k_fold_results.write('\n')
for i in range(10):
    k_fold_results.write(k_fold_lr[i])
    sum = sum+float(k_fold_lr[i])
    k_fold_results.write('\n')
sum= sum/10
k_fold_results.write('sum')
k_fold_results.write('\n')
k_fold_results.write(str(sum))

