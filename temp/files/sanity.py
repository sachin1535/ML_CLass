import os,glob,shutil
for f in glob.glob("*.txt"):
	num_lines = sum(1 for line in open(f))
        if num_lines < 20:
                src_path="/home/salman/ml_p/files/"+str(f)
		shutil.move(src_path,"/home/salman/ml_p/")


