from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import pandas as pd
import os
import glob

class dataProcess(object):

	def __init__(self, out_rows, out_cols, npy_path = "../npydata/",  num_class = 1):

		"""
		
		"""

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.npy_path = npy_path
		self.num_class = num_class
		


	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/img_train.npy")
		imgs_label_train = np.load(self.npy_path+"/img_train_label.npy")
		#imgs_train = imgs_train.astype('float32')
		#imgs_mask_train = imgs_mask_train.astype('float32')
		#imgs_train /= 255
		#mean = imgs_train.mean(axis = 0)
		#np.save(self.npy_path + '/imgs_train_mean.npy', mean)
		#imgs_train -= mean	
		return imgs_train,imgs_label_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/img_test.npy")
		#imgs_test = imgs_test.astype('float32')
		#imgs_test /= 255
		#mean = imgs_test.mean(axis = 0)
		#np.save(self.npy_path + '/imgs_test_mean.npy', mean)
		#imgs_test -= mean	
		return imgs_test




if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(256,256)
	mydata.create_train_data()
	mydata.create_test_data()
	imgs_train,imgs_label_train = mydata.load_train_data()
	print imgs_train.shape,imgs_label_train.shape
	print imgs_label_train[0]
	print imgs_train[0]