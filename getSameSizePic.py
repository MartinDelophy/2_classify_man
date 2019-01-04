from PIL import Image
import os
import shutil


'''
抽取图像文件
'''

baseDir = 'baseDir'

other = 'otherDir'

x = 0
y = 0
wx = 250
wy = 250


if not os.path.exists(other):
    os.makedirs(other)

for (dirpath,dirnames,filenames) in os.walk(baseDir):
    print (dirpath,dirnames,filenames)
    for filename in filenames :
        im = Image.open(dirpath + "/" + filename)
        region = im.crop((x, y, x+wx, y+wy))
        region.save(other + "/" + filename)

print ("done")





    