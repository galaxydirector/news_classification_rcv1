# train neural nets


from neural_nets import MLP, prepare_callbacks
import keras


# produce a list of parameters
param_list = []
for nb_layer in range(3):
	model_params = {'input_shape': (),
				'nb_filters': 100,
				'nb_layers': nb_layer,
				'output_classes': 2 }
	param_list.append(model_params)



# set up model
model = MLP(model_params).model
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
callback_list = prepare_callbacks(
		model_folder, 
		test_name, 
		input_shape,
		tr_params, 
		test_params)
model.fit_generator(generator, 
	steps_per_epoch=None, 
	epochs=5, 
	verbose=1, 
	callbacks=None)