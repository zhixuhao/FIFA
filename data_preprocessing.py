#coding=utf-8
import numpy as np
import skimage.io as io
import skimage.transform as trans
import sys,os
pathlist = []
pathlist_label = []

mode = "last4"
path = ""
if(mode == "last4"):
    path = "npydata/last4/"
else:
    path = "npydata/all7/"
#递归查找文件夹
def find_path(path):
    paths = os.listdir(path)
    for p in paths:
        if(p == "images"):
            pathlist.append(os.path.join(path,p))
            return
        if(os.path.isdir(os.path.join(path,p))):
            find_path(os.path.join(path,p))

find_path('../')
for p in pathlist:
    if p.find("比赛") > -1:
        print p,1
        pathlist_label.append(1)
    else:
        print p,0
        pathlist_label.append(0)

#use all videos, 70% as train data, 30% as test data
'''
img_train = []
img_train_label = []
img_test = []
img_test_label = []
for i in range(len(pathlist)):
    path = pathlist[i]
    label = pathlist_label[i]
    images = os.listdir(path)
    num = len(images)
    print path,num
    n = 0
    for name in images:
        img = io.imread(os.path.join(path,name))
        img = trans.resize(img,(192,256))
        img = img.astype('float32')
        if(n < num*0.7):
            img_train.append(img)
            img_train_label.append(label)
        else:
            img_test.append(img)
            img_test_label.append(label)
        n += 1
        if (n%200 == 0):
            print n
'''
#use last 4 videos as test data
for i in range(len(pathlist)):
    path = pathlist[i]
    label = pathlist_label[i]
    images = os.listdir(path)
    num = len(images)
    print path,num
    n = 0
    for name in images:
        img = io.imread(os.path.join(path,name))
        img = trans.resize(img,(192,256))
        img = img.astype('float32')
        if(i < len(pathlist)-4):
            img_train.append(img)
            img_train_label.append(label)
        else:
            img_test.append(img)
            img_test_label.append(label)
        n += 1
        if (n%200 == 0):
            print n

print len(img_train)
print len(img_train_label)   
print len(img_test)
print len(img_test_label)   
print len(img_train) + len(img_test)
img_train = np.array(img_train)
print img_train.shape
img_train = img_train.astype('float32')
np.save(path + 'img_train.npy',img_train)
img_train_label = np.array(img_train_label)
print img_train_label.shape
np.save(path + 'img_train_label.npy',img_train_label)
img_test_label = np.array(img_test_label)
print img_test_label.shape
np.save(path + 'img_test_label.npy',img_test_label)
img_test = np.array(img_test)
print img_test.shape
img_test = img_test.astype('float32')
np.save(path + 'img_test.npy',img_test)
'''
