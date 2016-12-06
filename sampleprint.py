# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import os
# random shuffle
import random
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import logging
import sys
import scipy.io

fw = open('tset.txt','w');
a = [1,2,3,4]
fw.write("Logistic Regression\n{}\n".format(a));
fw.close()