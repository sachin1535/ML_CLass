import os,glob,shutil
cnt=0;
for f in glob.glob("*.txt"):
       num_lines = sum(1 for line in open(f))
       if num_lines < 80:
               src_path="/home/sachin77/Documents/Sachin/word2vec-sentiment/ML_Project/"+str(f)
               shutil.move(src_path,"/home/sachin77/Documents/Sachin/word2vec-sentiment/Filter")
               cnt=cnt+1
               print(f)
print(cnt)
