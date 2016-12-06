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
# path = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/'
# pathTest = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/Test/'

#CHANGE THIS FILE NAME w.r.t analysis type
model = Doc2Vec.load('./imdb47.d2v')

log.info('Sentiment')
# CHANGE THESE VALUES BEFORE EACH ANALYSIS FOR DIFFERENT CLASSES 
# RUN PredictScore script to get these values
trainsamples = 99734;
testsamples = 235;
noOffeatures  =100;

train_arrays = numpy.zeros((trainsamples, noOffeatures))
train_labels = numpy.zeros(trainsamples)


#ALSO CHANGE THIS FILE IN MAIN FOLDER FROM THE DIFFERENT CLASSES FOLDERS 
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


############################################# 10 FOLD VALIDATION ########################################
# validationCnt = int(trainsamples*0.2);
# val_arrarys = numpy.zeros((validationCnt,noOffeatures))
# val_labels = numpy.zeros(validationCnt)
# train_arrays_val = numpy.zeros((trainsamples - validationCnt, noOffeatures))
# print(trainsamples - validationCnt)
# train_labels_val = numpy.zeros(trainsamples - validationCnt)
# validationScore_lr = list()
# validationScore_svm = list()
# validationScore_nn = list()

# for j in range(10):
#     index_shuf = range(len(train_arrays))
#     shuffle(index_shuf)
#     cnt=0;
#     for i in index_shuf:
#         if cnt < validationCnt:
#             val_arrarys[cnt] = train_arrays[i]
#             val_labels[cnt]  = train_labels[i]
#         else:
#             train_arrays_val[cnt-validationCnt] = train_arrays[i]
#             train_labels_val[cnt-validationCnt] = train_labels[i]
#         cnt = cnt+1

#     classifier  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001,multi_class = 'ovr')
#     classifier.fit(train_arrays_val, train_labels_val)
#     print classifier.score(val_arrarys, val_labels)
#     validationScore_lr.append(classifier.score(val_arrarys, val_labels))
    
# ## NN Classifier

#     clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(200,),random_state=1)
#     clf.fit(train_arrays_val,train_labels_val)
#     validationScore_nn.append(clf.score(val_arrarys, val_labels))
#     print clf.score(val_arrarys, val_labels)


# ## SVM Classifier
#     svm = SVC(C = 100,gamma = 'auto',kernel ='rbf')
#     svm.fit(train_arrays_val,train_labels_val)
#     validationScore_svm.append(svm.score(val_arrarys, val_labels))
#     print svm.score(val_arrarys, val_labels)

    

# print(validationScore_lr)
# print ("NN Classifier")
# print(validationScore_nn)
# print ("SVM Classifier")
# print(validationScore_svm)

# #CHANGE FILE NAME
# fw = open('validation_Result_28.txt','w');
# fw.write("Logistic Regression\n{}\n \nNN\n{} \nSVM \n{}\n".format(validationScore_lr,validationScore_nn,validationScore_svm));
# fw.close()

########################################################################################################

################################ TRAINING ERROR ########################################################
validationScore_lr = list()
validationScore_svm = list()
validationScore_nn = list()

#Validation data computation 
ratio =0.2 
validationCnt = int(trainsamples*(ratio1));
val_arrarys = numpy.zeros((validationCnt,noOffeatures))
val_labels = numpy.zeros(validationCnt)
train_arrays_val1 = numpy.zeros((trainsamples - validationCnt, noOffeatures))
print(trainsamples - validationCnt)
train_labels_val1 = numpy.zeros(trainsamples - validationCnt)
index_shuf = range(len(train_arrays))
shuffle(index_shuf)
cnt=0;
for i in index_shuf:
if cnt < validationCnt:
    val_arrarys[cnt] = train_arrays[i]
    val_labels[cnt]  = train_labels[i]
else:
    train_arrays_val1[cnt-validationCnt] = train_arrays[i]
    train_labels_val1[cnt-validationCnt] = train_labels[i]
cnt = cnt+1
trainsamples = range(len(train_arrays_val1))
for ratio in range(1,10):
    ratio1 = ratio/10.0;
    print(ratio1)
    trainCnt = int(trainsamples*(ratio1));
    
    train_arrays_val = numpy.zeros((trainCnt, noOffeatures))
    print(trainsamples - trainCnt)
    train_labels_val = numpy.zeros(trainCnt)

    # SHUFFLING THE DATA BEFORE DIVIDING INTO VALIDATION AND TRAIN
    index_shuf = range(len(train_arrays_val1))
    shuffle(index_shuf)
    cnt=0;
    for i in index_shuf:
        if cnt < trainCnt:
            train_arrays_val[cnt] = train_arrays[i]
            train_labels_val[cnt]  = train_labels[i]
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
    svm = SVC(C = 100,gamma = 'auto',kernel ='rbf')
    svm.fit(train_arrays_val,train_labels_val)
    validationScore_svm.append(svm.score(val_arrarys, val_labels))
    print svm.score(val_arrarys, val_labels)

print(validationScore_lr)
print ("NN Classifier")
print(validationScore_nn)
print ("SVM Classifier")
print(validationScore_svm)

#CHANGE FILE NAME
fw = open('training_Result_47.txt','w');
fw.write("Logistic Regression\n{}\n \nNN\n{} \nSVM \n{}\n".format(validationScore_lr,validationScore_nn,validationScore_svm));
fw.close()





####################################################################################################
print("training is complete");

test_arrays = numpy.zeros((testsamples, noOffeatures))
test_labels = numpy.zeros(testsamples)
# REMEMBER TO CHANGE THIS FILE BEFORE RUNNING CODE in current folder 
fd = open("detailInfo.txt",'r');
print(cnt);
cnt = 0;
cntc= 0;
print(classRecords);
for line in fd:
    cntc=0;
    parts = line.split('\t');
    classCnt = classRecords[parts[1]];
    for each_sample in range(5):
        sample_category = "TEST_"+parts[1]+"_" + str(cntc)
        cntc = cntc+1;
        test_arrays[cnt] = model.docvecs[sample_category]
        test_labels[cnt] = classCnt;
        cnt = cnt+1;

fd.close();

print(test_labels);
log.info('Fitting')

# TEST SCORE COMPUTATION
test_array_score_lr = list()
test_array_score_nn = list()
test_array_score_svm = list()
classifier  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
      intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001,multi_class = 'ovr')
classifier.fit(train_arrays, train_labels)
print classifier.score(test_arrays, test_labels)
test_array_score_lr.append(classifier.score(test_arrays, test_labels))

## NN Classifier

clf=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(200,),random_state=1)
clf.fit(train_arrays_val,train_labels_val)
test_array_score_nn.append(clf.score(test_arrays, test_labels))
print clf.score(test_arrays, test_labels)


## SVM Classifier
svm = SVC(C = 100,gamma = 'auto',kernel ='rbf')
svm.fit(train_arrays_val,train_labels_val)
test_array_score_svm.append(svm.score(test_arrays, test_labels))
print svm.score(test_arrays, test_labels)

#CHANGE FILE NAME 
fw = open('test_result_47.txt','w');
fw.write("Logistic Regression\n{}\n \nNN\n{} \nSVM \n{}\n".format(test_array_score_lr,test_array_score_nn,test_array_score_svm));
fw.close()
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
