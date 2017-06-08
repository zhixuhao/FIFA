import numpy as np
import skimage.io as io

path = ['/home/zhixuhao/Documents/FIFA/npydata/','/home/zhixuhao/Documents/FIFA/npydata/all_7split/']
pathr = '/home/zhixuhao/Documents/FIFA/FIFA/'
path_dir = ['/home/zhixuhao/Documents/FIFA/analysis/last4/','/home/zhixuhao/Documents/FIFA/analysis/all7/']
res = ['out_last4.npy','out_all7.npy']

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
                io.imsave(path_dir[i] + str(j) + '.jpg',img)
                count += 1
    print "count: ",count    
      
