import os; 
path = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/'
pathTest = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/Test/'
fd = open("detailInfo.txt",'r');
sum = 0;
for line in fd:
	parts = line.split('\t');
	#print(int(parts[2][:-1]));
	sum = sum + int(parts[2][:-1]);
print(sum);
fd.close();
testsum = 0;
for fn in os.listdir(pathTest):
	if fn.endswith('.txt'):
		ftrain = open(pathTest+fn,'r');
		sentencesList = ftrain.readlines()
		count = len(sentencesList);
		print(count);
		testsum = testsum +count;
		ftrain.close();
print(testsum)