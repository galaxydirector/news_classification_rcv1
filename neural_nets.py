# neural nets

import os
import keras
from glob import glob
# import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Input, merge, Dense, BatchNormalization, Activation, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard


class MLP(object):
	def __init__(self, model_params):
		x, y = self.build_model(**model_params)
		self.model = Model(inputs=x, outputs=y)

	def build_model(self,
					  input_shape,
					  nb_filters,
					  nb_layers,
					  use_bias=True):

		print("Input shape is ",input_shape)
		input = Input(shape=input_shape, name='input_layer')
		out = input

		for _ in range(nb_layers):
			out = Dense(nb_filters, activation='relu')(out)

		# out = Dense(output_classes, activation='softmax')(out)
		out = Dense(1)(out)

		return input, out

def prepare_callbacks(model_folder, test_name, use_adaptive_optimzer=True):
	
	model_directory = os.path.join(*[model_folder,test_name])
	if not os.path.exists(model_directory):
	    os.makedirs(model_directory)

	saved_model_path = os.path.join(*[model_directory,"weights_{epoch:05d}.hdf5"])
	csv_log_path = os.path.join(*[model_directory,"{}_log.csv".format(test_name)])

	save_chkpt = ModelCheckpoint(
		saved_model_path,
		verbose=1,
		save_best_only=False,
		monitor='loss',
		mode='auto',
		period=5
	)

	tr_logger = CSVLogger(csv_log_path, separator=',', append=True)

	if use_adaptive_optimzer:
		reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1, mode="min")
		callback_list = [reduce_lr, save_chkpt, tr_logger]
	else:
		callback_list = [save_chkpt, tr_logger]
	
	return callback_list