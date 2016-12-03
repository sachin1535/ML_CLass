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
log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')
trainsamples = 99253;
testsamples = 175;
noOffeatures  =100;

train_arrays = numpy.zeros((trainsamples, noOffeatures))
train_labels = numpy.zeros(trainsamples)
path = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/'
pathTest = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/Test/'
fd = open("detailInfo.txt",'r');
classCnt = 0;
classRecords = dict();
cnt = 0;
cntc= 0;


for line in fd:
    cntc = 0;
    parts = line.split('\t');
    classRecords[parts[1]] = classCnt;
    for each_sample in range(int(parts[2][:-1])):
        sample_category = parts[1]+"_" + str(cntc)
        cntc = cntc+1;
        print(sample_category);
        train_arrays[cnt] = model.docvecs[sample_category]
        train_labels[cnt] = classCnt;
        cnt = cnt+1;
    classCnt = classCnt+1;
fd.close();
print numpy.max(train_labels)

validationCnt = int(trainsamples*0.2);
val_arrarys = numpy.zeros((validationCnt,noOffeatures))
val_labels = numpy.zeros(validationCnt)
train_arrays_val = numpy.zeros((trainsamples - validationCnt, noOffeatures))
print(trainsamples - validationCnt)
train_labels_val = numpy.zeros(trainsamples - validationCnt)
validationScore_lr = list()
validationScore_svm = list()
validationScore_nn = list()

for j in range(10):
    index_shuf = range(len(train_arrays))
    shuffle(index_shuf)
    cnt=0;
    for i in index_shuf:
        if cnt < validationCnt:
            val_arrarys[cnt] = train_arrays[i]
            val_labels[cnt]  = train_labels[i]
        else:
            train_arrays_val[cnt-validationCnt] = train_arrays[i]
            train_labels_val[cnt-validationCnt] = train_labels[i]
        cnt = cnt+1

    classifier  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001,multi_class = 'ovr')
    classifier.fit(train_arrays_val, train_labels_val)
    print classifier.score(val_arrarys, val_labels)
    validationScore_lr.append(classifier.score(val_arrarys, val_labels))
    
## NN Classifier

    clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(200,),random_state=1)
    clf.fit(train_arrays_val,train_labels_val)
    validationScore_nn.append(clf.score(val_arrarys, val_labels))
    print clf.score(val_arrarys, val_labels)


## SVM Classifier
    svm = SVC(C = 1000000,gamma = 'auto',kernel ='rbf')
    svm.fit(train_arrays_val,train_labels_val)
    validationScore_svm.append(svm.score(val_arrarys, val_labels))
    print svm.score(val_arrarys, val_labels)

    

print(validationScore_lr)
print ("NN Classifier")
print(validationScore_nn)
print ("SVM Classifier")
print(validationScore_svm)



# fd = open("detailInfo.txt",'r');
# print("trainin is complete");
# print(cnt);
# cnt = 0;
# cntc= 0;
# print(classRecords);
# for line in fd:
#     cntc=0;
#     parts = line.split('\t');
#     classCnt = classRecords[parts[1]];
#     for each_sample in range(5):
#         sample_category = "TEST_"+parts[1]+"_" + str(cntc)
#         cntc = cntc+1;
#         test_arrays[cnt] = model.docvecs[sample_category]
#         test_labels[cnt] = classCnt;
#         cnt = cnt+1;

# fd.close();
# print(cnt);
# cntc=0;
# for i in range(5):
#     sample_category = "TEST_"+parts[1]+"_" + str(cntc)
#     cntc = cntc+1;
#     test_arrays[cnt] = model.docvecs[sample_category]
#     test_labels[cnt] = classRecords["MONEY"];
#     cnt = cnt+1;

# print(classRecords["MONEY"]);  
# print(test_labels);
# log.info('Fitting')


# #LR 
# classifier  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001,multi_class = 'ovr')
# classifier.fit(train_arrays, train_labels)
# print classifier.score(test_arrays, test_labels)

# #SVM
# cld=svm.SVC()

# cld.fit(train_arrays,train_labels)
# print cld.score(test_arrays,test_labels)

'''
labels = classifier.predict(test_arrays)
print(len(labels));

for i in range(len(test_labels)-5,len(test_labels)): 
    if test_labels[i]==labels[i]:
        print("correct");
#print labels

print(cntc);
'''
