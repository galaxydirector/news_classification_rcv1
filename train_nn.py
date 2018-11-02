# train neural nets


from neural_nets import MLP, prepare_callbacks
import keras

model_folder = "/home/aitrading/Desktop/google_drive/Course_Work/ESE545/Project2/models"

# produce a list of parameters
# iterate three nb_layers
param_list = []
for nb_layer in range(3): # no more than 6 layers
	model_params = {'input_shape': (),
				'nb_filters': 100,
				'nb_layers': nb_layer,
				'output_classes': 2 }
	param_list.append(model_params)

	test_name = '{}layers'.format(nb_layer)

# set up model
model = MLP(model_params).model
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
callback_list = prepare_callbacks(
		model_folder, 
		test_name)
# model.fit(data, 
# 	one_hot_labels, 
# 	epochs=10, 
# 	verbose=1, 
# 	batch_size=32,
# 	callbacks = callback_list)
generator = dense_data_generator(train_x,train_y,T)

model.fit_generator(generator, 
	steps_per_epoch=None, # set at generator T iterations
	epochs=5, 
	verbose=1, 
	callbacks=None,
	max_queue_size=32, 
	workers=3, 
	use_multiprocessing=False)

####################################################
# produce a list of parameters
# iterate three nb_layers
param_list = []
filter_list = [50,100,150,300,500]

for nb_filters in filter_list: # no more than 6 layers
	model_params = {'input_shape': (),
				'nb_filters': nb_filters,
				'nb_layers': 5,
				'output_classes': 2 }
	param_list.append(model_params)


score = model.evaluate(x_test, y_test, batch_size=128)