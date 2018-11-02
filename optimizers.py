# PEGASOS ALGO
# ADAGRAD ALGO
import numpy as np
from data_import import *
import tensorflow as tf
# from accuracy import *


def get_accuracy(x,y,w):
	'''
	x: m x d data as csr_matrix
	y: m x 1 label as csr_matrix
	w: d x 1 weights as numpy array
	'''

	return round(int(sum(x.multiply(w.reshape(-1)).sum(1)>0))/x.shape[0],2)


def PEGASOS_old(train_x,train_y,T,lamb,test_x,test_y):
	# lamb would be chosen from 0.0001, 0.001, 0.01, 0.1, 1, 10, 100

	# initialize a w with normal of std = 0.01
	w = 0.01*np.random.randn(train_x.shape[1],1)/np.sqrt(lamb) # start with very small numbers with mean of 0
	acc = get_accuracy(test_x,test_y,w)
	print("Current test accuracy is: "+str(acc))
	# initialize the generator
	data_gen = dense_data_generator(train_x,train_y,T)
	for i in range(1,T+1):
		sub_next = next(data_gen)
		sub_x = sub_next[0]
		sub_y = sub_next[1]
		# note: numpy matmul is equivalent to dot
		# while numpy multipy is equivalent to *, which is element wise
		output = np.matmul(sub_x,w)*sub_y
		output_ind = output<1
		output_ind = output_ind.reshape(-1)
		total_loss = np.sum(np.multiply(sub_y[output_ind],sub_x[output_ind]),axis=0)
		total_loss = total_loss.reshape((total_loss.shape[0],1))
		gradient = lamb*w - np.divide(total_loss,sub_x.shape[0]) 
		lr = 1/(i*lamb)
		raw_w = w - lr*gradient
		w = np.minimum(1,1/np.sqrt(lamb)/np.sqrt(sum([j**2 for j in raw_w])))*raw_w
		
		print(str(i)+" Iterations Completed.")

		if i % 50 ==0:
			acc = get_accuracy(test_x,test_y,w)
			print("Current test accuracy is: "+str(acc))

	acc = get_accuracy(test_x,test_y,w)
	print("Last test accuracy is: "+str(acc))
	
	return w




def PEGASOS(data_gen, lamb, test_x, test_y, n_features):
	# lamb would be chosen from 0.0001, 0.001, 0.01, 0.1, 1, 10, 100

	# initialize a w with normal of std = 0.01
	w = 0.01*np.random.randn(n_features,1)/np.sqrt(lamb) # start with very small numbers with mean of 0
	# acc = get_accuracy(test_x,test_y,w)
	# print("Current test accuracy is: "+str(acc))
	
	# data_gen = dense_data_generator(train_x,train_y,T)
	# for i in range(1,T+1):
	# 	sub_next = next(data_gen)
	# 	sub_x = sub_next[0]
	# 	sub_y = sub_next[1]

	for i, (sub_x,sub_y) in enumerate(data_gen,1):

		# note: numpy matmul is equivalent to dot
		# while numpy multipy is equivalent to *, which is element wise
		output = np.matmul(sub_x,w)*sub_y
		output_ind = output<1
		output_ind = output_ind.reshape(-1)
		total_loss = np.sum(np.multiply(sub_y[output_ind],sub_x[output_ind]),axis=0).reshape(-1,1)
		gradient = lamb*w - np.divide(total_loss,sub_x.shape[0])
		lr = 1/(i*lamb)
		raw_w = w - lr*gradient
		w = np.minimum(1,1/np.sqrt(lamb)/np.sqrt(sum([j**2 for j in raw_w])))*raw_w
		
		print(str(i)+" Iterations Completed.")

		# if i % 50 ==0:
		# 	acc = get_accuracy(test_x,test_y,w)
		# 	print("Current test accuracy is: "+str(acc))

	acc = get_accuracy(test_x,test_y,w)
	print("Last test accuracy is: "+str(acc))
	
	return w












def adagrad(w, prediction, target, lr):
	"""
	lr = 1/sqrt(T)
	"""

	return






# # initialize the generator
# data_gen = dense_data_generator(train_x,train_y,T)


def PEGASOS_tf(data_gen, input_lamb, n_features): #################################################### developing
	# tensorflow version of PEGASOS algorithm


	# initialize weights with normal of std = 0.01, mean = 0
	# w = tf.variables(tf.random_normal([tf.shape(x)[1],1],stddev=0.01)/tf.sqrt(lamb), name='weights')
	w = tf.Variable(tf.random_normal([n_features,1],stddev=0.01)/tf.sqrt(input_lamb), name='weights')

	# row_per_update = tf.floor_div(tf.shape(x)[0], T)

	# sub_x = tf.placeholder(tf.float32, shape=[row_per_update, tf.shape(x)[1]])
	# sub_y = tf.placeholder(tf.float32, shape=[row_per_update, tf.shape(y)[1]])
	sub_x = tf.placeholder(tf.float32, shape=None)
	sub_y = tf.placeholder(tf.float32, shape=None)
	ite = tf.placeholder(tf.float32, shape=None)
	lamb = tf.constant(input_lamb)
	# output = np.matmul(sub_x,w)*sub_y
	# output_ind = output<1
	# output_ind = output_ind.reshape(-1)
	# total_loss = np.sum(np.multiply(sub_y[output_ind],sub_x[output_ind]),axis=0).reshape(-1,1)
	# gradient = lamb*w - np.divide(total_loss,sub_x.shape[0]) 






	output = tf.multiply(tf.matmul(sub_x,w),sub_y)
	output_ind = tf.reshape(tf.less(output,1),[-1])
	total_loss = tf.reshape(tf.reduce_sum(tf.multiply(tf.boolean_mask(sub_y,output_ind),tf.boolean_mask(sub_x,output_ind)), axis=0),[-1,1])
	gradient = tf.multiply(lamb,w) - tf.divide(total_loss,tf.to_float(tf.shape(sub_x)[0])) 

	# lr = 1/(tf.to_float(ite)*lamb)
	lr = 1/(ite*lamb)

	raw_w = w - lr*gradient
	update_w = w.assign(tf.minimum(1.0,1.0/tf.sqrt(lamb)/tf.norm(raw_w))*raw_w)

	# tf.sqrt(sum([i**2 for i in raw_w])) # is norm?

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		# for i in range(T):
			# new_w = sess.run(update_w, feed_dict = {sub_x: x[i:(i+row_per_update),:],sub_y: y[i:(i+row_per_update),:],ite:i})
		for i, (x,y) in enumerate(data_gen,1):
			new_w = sess.run(update_w, feed_dict = {sub_x: x, sub_y: y, ite:i})
			print(str(i)+" Iterations Completed.")
			
	return new_w


if __name__ == '__main__':
	from data_import import * 

	csr_data = import_data()
	data_gen = dense_data_generator(csr_data, 10000)
	w = PEGASOS(data_gen, lamb=0.01, n_features = csr_data.get_shape()[1]-1)

	print(w)