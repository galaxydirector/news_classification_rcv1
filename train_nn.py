# train neural nets
from neural_nets import MLP, prepare_callbacks
import keras
from data_import import *

model_folder = "/home/aitrading/Desktop/google_drive/Course_Work/ESE545/Project2/models"
sparse_data = import_data()
train_x, train_y, test_x, test_y = data_set_sparse(sparse_data)

samples_per_batch = 2000
iterations = 100000// samples_per_batch
# produce a list of parameters
# iterate three nb_layers
# param_list = []
# nb_filters = 100
# for nb_layer in range(1,7): # no more than 6 layers
# 	model_params = {'input_shape': (1,train_x.shape[1]),
# 				'nb_filters': nb_filters,
# 				'nb_layers': nb_layer}
# 	test_name = '{}layers_{}filters'.format(nb_layer,nb_filters)

param_list = []
filter_list = [50,100,150,300,500]
nb_layer = 4
for nb_filters in filter_list: # no more than 6 layers
	model_params = {'input_shape': (1,train_x.shape[1]),
				'nb_filters': nb_filters,
				'nb_layers': nb_layer}
	# param_list.append(model_params)


	test_name = '{}filters{}layers_'.format(nb_filters,nb_layer)

	# set up model
	model = MLP(model_params).model
	model.compile(optimizer='adagrad',
	              loss='mean_squared_error',
	              metrics=['accuracy'])
	callback_list = prepare_callbacks(
			model_folder, 
			test_name)
	# generator = dense_data_generator(train_x,train_y,iterations)

	model.fit_generator(generator=dense_data_generator(train_x,train_y,iterations), 
		steps_per_epoch=iterations, 
		epochs=5, 
		verbose=1, 
		callbacks=callback_list,
		validation_data=dense_data_generator(test_x,test_y,7000),
		validation_steps=20,
		max_queue_size=3,
		use_multiprocessing=False)

# ####################################################
# # produce a list of parameters
# # iterate three nb_layers
# param_list = []
# filter_list = [50,100,150,300,500]

# for nb_filters in filter_list: # no more than 6 layers
# 	model_params = {'input_shape': (),
# 				'nb_filters': nb_filters,
# 				'nb_layers': 5,
# 				'output_classes': 2 }
# 	param_list.append(model_params)


# score = model.evaluate(x_test, y_test, batch_size=128)


if __name__ == '__main__':
	main()