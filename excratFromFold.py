import os
import shutil

baseDir = './lfw'
man = './man'
if not os.path.exists(man):
      os.makedirs(man)



for (dirpath,dirnames,filenames) in os.walk(baseDir):
    for filename in filenames :
        shutil.move(dirpath + "/" + filename, man + "/" + filename)

print ("done")
        