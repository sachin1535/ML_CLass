import os; 
path = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/'
pathTest = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/Test/'
fd = open("detailInfo.txt",'r');
sum = 0;
cnt =0;
for line in fd:
	parts = line.split('\t');
	#print(int(parts[2][:-1]));
	sum = sum + int(parts[2][:-1]);
	cnt=cnt+1
print("trainsamples{}\t".format(sum));
fd.close();
print("testSamples{}\t".format(cnt*5))
testsum = 0;
for fn in os.listdir(pathTest):
	if fn.endswith('.txt'):
		ftrain = open(pathTest+fn,'r');
		sentencesList = ftrain.readlines()
		count = len(sentencesList);
		# print(count);
		testsum = testsum +count;
		ftrain.close();
# print(testsum)