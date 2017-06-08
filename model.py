import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from data import dataProcess
from keras import backend as K
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return 1.-dice_coef(y_true, y_pred)


class multiNet(object):

	def __init__(self, img_rows = 192, img_cols = 256, label_num = 1):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.label_num = label_num

	def load_train_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_label_train = mydata.load_train_data()
		return imgs_train, imgs_label_train

	def load_test_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_test,imgs_label_test = mydata.load_test_data()
		return imgs_test,imgs_label_test

	

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


	def train(self):

		print("loading data")
		imgs_train, train_label = self.load_train_data()
		imgs_test, imgs_test_label = self.load_test_data()
		print("loading data done")
		model = self.get_model()
		print("got multinet")

		model_checkpoint = ModelCheckpoint('multinet_all7.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		history = model.fit(imgs_train, train_label, batch_size=64, nb_epoch=20, validation_data=(imgs_test,imgs_test_label), verbose=1, shuffle=True, callbacks=[model_checkpoint])
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

		print("loading data")
		imgs_test, imgs_test_label = self.load_test_data()
		print("loading data done")

		model = self.get_model()
		model.load_weights('multinet_all7.hdf5')
		print('predict test data')
		out = model.predict(imgs_test, batch_size=64, verbose=1)
		out = out[:,0]
		out[out > 0.5] = 1
		out[out < 0.5] = 0
		error = out - imgs_test_label
		sum_error = np.sum(np.abs(error))
		np.save('out_all7.npy', out)
		eva = model.evaluate(imgs_test,imgs_test_label,batch_size=64, verbose=1)
		print "eva:",eva
		print "error num:",sum_error," total num:",imgs_test_label.shape



if __name__ == '__main__':
	mynet = multiNet()
	#model = mynet.get_model()
	#mynet.train()
	mynet.test()
