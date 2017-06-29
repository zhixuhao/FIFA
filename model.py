#coding=utf-8
'''
train with img train and 800X600,1024X768 image,test with img test and 800X600,1024X768,1440X900
acc = 99%
'''
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from data import dataProcess
from keras import backend as K
from sklearn.metrics import matthews_corrcoef
#import matplotlib.pyplot as plt
import analysis
import skimage.io as io

def dice_coef(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return 1.-dice_coef(y_true, y_pred)


class multiNet(object):

	def __init__(self, img_rows = 192, img_cols = 256, label_num = 1, mode = "all7", small = True):

		'''
		
		'''
		print "mode is ",mode
		self.img_rows = img_rows
		self.img_cols = img_cols
		self.label_num = label_num
		self.mode = mode
		self.threshold = 0.5
		self.model_txt = ""
		self.small = small
		if(small):
			self.model_txt = "small"
		if(mode == "last4"):
			self.path_dir = "analysis/last4/"
			self.weight = "multinet_last4"+self.model_txt+".hdf5"
			self.res = "out_last4"+self.model_txt+".npy"
		if(mode == "extra"):
			
			self.path_dir = "analysis/extra/"
			self.weight = "multinet_extra0629"+self.model_txt+".hdf5"
			self.res = "out_extra"+self.model_txt+".npy"
		if(mode == "all7"):
			self.path_dir = "analysis/all7/"
			self.weight = "multinet_all7"+self.model_txt+".hdf5"
			self.res = "out_all7"+self.model_txt+".npy"
			

	def load_train_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols, mode = self.mode)
		imgs_train, imgs_label_train = mydata.load_train_data()
		return imgs_train, imgs_label_train

	def load_test_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols, mode = self.mode)
		imgs_test,imgs_label_test = mydata.load_test_data()
		return imgs_test,imgs_label_test

	def add_extra_data(self, pre_data, pre_label, path, arr):
		
		for item in arr:
			print "load",item
			tmp_data = np.load(os.path.join(path,item+".npy"))
			tmp_label = np.load(os.path.join(path,item+"_label.npy"))
			print "load done",item
			pre_data = np.concatenate((pre_data,tmp_data),axis=0)
			pre_label = np.concatenate((pre_label,tmp_label),axis=0)
		return pre_data,pre_label
	

	def get_model(self):
		
		'''
		using vgg-16
		'''

		inputs = Input((self.img_rows, self.img_cols,3))

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		#conv4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		#conv5 = Dropout(0.5)(conv5)
		pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

		pool5 = Flatten()(pool5)

		fc6 = Dense(4096, activation = 'relu')(pool5)
		fc6 = Dropout(0.5)(fc6)

		fc7 = Dense(4096, activation = 'relu')(fc6)
		fc7 = Dropout(0.5)(fc7)

		fc8 = Dense(self.label_num, activation = 'sigmoid')(fc7)			

		model = Model(input = inputs, output = fc8)
		#model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics=[dice_coef,distance_loss])
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])																																												

		return model
	
	def get_small_model(self):
		
		inputs = Input((self.img_rows, self.img_cols,3))

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		#conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		#conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		#conv4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		#conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		#conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		#conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		#conv5 = Dropout(0.5)(conv5)
		#pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

		pool5 = Flatten()(pool4)

		fc6 = Dense(512, activation = 'relu')(pool5)
		fc6 = Dropout(0.5)(fc6)

		fc7 = Dense(512, activation = 'relu')(fc6)
		fc7 = Dropout(0.5)(fc7)

		fc8 = Dense(self.label_num, activation = 'sigmoid')(fc7)			

		model = Model(input = inputs, output = fc8)
		#model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics=[dice_coef,distance_loss])
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])																																												

		return model


	def train(self):

		'''
		you can reduce the batch_size if the memory is not enough.

		nb_epoch is the total num of epoch during training process.
		'''

		print("loading data")
		imgs_train, train_label = self.load_train_data()
		imgs_test, imgs_test_label = self.load_test_data()
		
		imgs_train, train_label = self.add_extra_data(imgs_train,train_label,"../npydata/0609/npydata/",["800X600_Full_hall_0","1024X768_Full_hall_0"])
		imgs_test, imgs_test_label = self.add_extra_data(imgs_test,imgs_test_label,"../npydata/0609/npydata/",["800X600_Full_hall_1","1024X768_Full_hall_1"])
		imgs_train, train_label = self.add_extra_data(imgs_train,train_label,"../npydata/0629/npydata/",["800X600_Win_hall","800X600_Win_game","1024X768_Win_hall","1024X768_Win_game"])
		imgs_test, imgs_test_label = self.add_extra_data(imgs_test,imgs_test_label,"../npydata/0629/npydata/",["1280X720_Win_hall","1280X720_Win_game","720X576_Win_hall","720X576_Win_game"])
		print("loading data done")
		if(self.small):
			model = self.get_small_model()
		else:
			model = self.get_model()
		print("got multinet")

		model_checkpoint = ModelCheckpoint(self.weight, monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		history = model.fit(imgs_train, train_label, batch_size=64, nb_epoch=8, validation_data=(imgs_test,imgs_test_label), verbose=1, shuffle=True, callbacks=[model_checkpoint])
		'''
		print(history.history.keys())
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
		'''


	def test(self):

		'''
		predict and analyze test data
		'''

		print("loading data")
		imgs_test, imgs_test_label = self.load_test_data()
		if(self.mode == "extra"):
			imgs_test,imgs_test_label = self.add_extra_data(imgs_test,imgs_test_label,"../npydata/0609/npydata",["800X600_Full_hall_1","1024X768_Full_hall_1","1440X900_Full_hall_1"])
		print("loading data done")
		if(self.small):
			model = self.get_small_model()
		else:
			model = self.get_model()
		model.load_weights(self.weight)
		print('predict test data')
		out = model.predict(imgs_test, batch_size=64, verbose=1)
		out = out[:,0]
		out[out > self.threshold] = 1
		out[out < self.threshold] = 0
		error = out - imgs_test_label
		sum_error = np.sum(np.abs(error))
		#np.save(self.res, out)
		eva = model.evaluate(imgs_test,imgs_test_label,batch_size=64, verbose=1)
		print "eva:",eva
		print "error num:",sum_error," total num:",imgs_test_label.shape
		#self.analyze(imgs_test, imgs_test_label, out)

	def analyze(self, test_img, test_label, test_res):
		count = 0
		for j in range(len(test_label)):
			if(test_label[j] != test_res[j]):
				print "image",j
				img = test_img[j,:,:,:]
				t = ""
				if(test_label[j] == 1):
					t = "game"
				if(test_label[j] == 0):
					t = "hall"
				io.imsave(self.path_dir + str(j) + "_" + t + '.jpg',img)
				count += 1
		print "count: ",count  

	def test_one(self, model, name):
		print "loading ",name
		imgs_test = np.load(os.path.join('../npydata/0629/npydata',name+'.npy'))
		imgs_test_label = np.load(os.path.join('../npydata/0629/npydata',name+'_label.npy'))
		out = model.predict(imgs_test, batch_size=64, verbose=1)
		out = out[:,0]
		out[out > self.threshold] = 1
		out[out < self.threshold] = 0
		error = out - imgs_test_label
		sum_error = np.sum(np.abs(error))
		eva = model.evaluate(imgs_test,imgs_test_label,batch_size=64, verbose=1)
		print name,"eva:",eva
		print "error num:",sum_error," total num:",imgs_test_label.shape
		
	def test_0629(self):
		model = self.get_small_model()
		model.load_weights('multinet_extra0629small.hdf5')
		test_arr = []
		tmp_arr = os.listdir('../npydata/0629/npydata')
		for t in tmp_arr:
			if(t.find('_label') > -1):
				test_arr.append(t[:len(t)-10])
		for n in test_arr:
			self.test_one(model,n)

if __name__ == '__main__':
	
	'''
	mode is either last4 or all7

	get_model() returns a CNN-based VGG-16 model, designed for image classification

	train() will train this model by fitting to the training data

	test() will predict the testing data, and generate a predicting result as a .npy file

	analyze() will pick all the error predictions, and save them as images, with postfix '比赛' 
	or '大厅', which is the true label.

	'''
	
	mynet = multiNet(mode="extra", small=True)
	#mynet.train()
	mynet.test()
