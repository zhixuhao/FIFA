#coding=utf-8
import numpy as np
import skimage.io as io

path = ['npydata/last4','npydata/all7/']
pathr = ''
path_dir = ['analysis/last4/','analysis/all7/']
res = ['out_last4.npy','out_all7.npy']

def analyze():
    for i in range(2):
        print "loading data..."
        test_img = np.load(path[i] + 'img_test.npy')
        test_label = np.load(path[i] + 'img_test_label.npy')
        test_res = np.load(pathr + res[i])
        count = 0
        for j in range(len(test_label)):
              if(test_label[j] != test_res[j]):
                    print "image",j
                    img = test_img[j,:,:,:]
                    t = ""
                    if(test_label[j] == 1):
                        t = "比赛"
                    if(test_label[j] == 0):
                        t = "大厅"
                    io.imsave(path_dir[i] + str(j) + "_" + t + '.jpg',img)
                    count += 1
        print "count: ",count    
      
