import os;
path = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/'
testPath = '/home/sachin77/Documents/Sachin/word2vec-sentiment/filterStop/Test/'
sources = dict()
fd = open('detailInfo.txt','w');
testCount  = 5
for fn in os.listdir(path):
	if fn.endswith('.txt'):
		ftrain = open(path+fn,'r');
		sentencesList = ftrain.readlines()
		count = len(sentencesList);
		fileparts = (fn.split('.'))[0]
		ftrain.close()
		
		if (count-testCount) > 0:
			ftrain = open(path+fn,'w');
			ftest = open(testPath+"TEST_"+fileparts.upper()+".txt",'w'); 
			print(sentencesList[count-testCount:count]);		
			for line in sentencesList[count-testCount:count]:
				ftest.write("{}".format(line));
			ftest.close();
			for line in sentencesList[0:count-testCount]:
				ftrain.write("{}".format(line));
			count = count -testCount;
			fd.write("{}\t{}\t{}\n".format(str(path+fn),fileparts.upper(),count));
			ftrain.close()
		sources[str(path+fn)] = str(fileparts.upper());
fd.close();
