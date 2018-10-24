# neural nets

# import os.path
import keras
# import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Input, merge, Dense, BatchNormalization, Activation, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

model_params = {'input_shape': (),
				'nb_filters': 100,
				'nb_layers': nb_layer,
				'output_classes': 2 }

class MLP(object):
    def __init__(self,
		    	**model_params):

		x, y = self.build_model(**model_params)
		self.model = Model(inputs=x, outputs=y)


	def build_model(self,
					  input_shape,
					  nb_filters,
					  nb_layers,
					  output_classes,
					  use_bias=True):

		print("Input shape is ",input_shape)
		input = Input(shape=input_shape, name='input_layer')
		out = input

		for _ in range(nb_layers):
			out = Dense(nb_filters, activation='relu')(out)

		out = Dense(output_classes, activation='softmax')(out)

		return input, out