# PEGASOS ALGO
# ADAGRAD ALGO
import numpy as np


def PEGASOS(x,y,T,lamb):
	# lamb would be chosen from 0.0001, 0.001, 0.01, 0.1, 1, 10, 100

	# initialize a w with normal of std = 0.01
	w = 0.01*np.random.randn(x.shape[1],1)/np.sqrt(lamb) # start with very small numbers with mean of 0
	row_per_update = x.shape[0] // T

	for i in range(T):
		# what does a random subset means?
		# does that mean a element could be drawn more than once?
		# 1. if not allow to drawn more than once, then shuffle before enter (implement in this version)
		# 2. otherwise, np.random.randint(0,size) then take the elements

		sub_x = x[i:(i+row_per_update),:]
		sub_y = y[i:(i+row_per_update),:]

		# note: numpy matmul is equivalent to dot
		# while numpy multipy is equivalent to *, which is element wise
		output = np.matmul(sub_x,w)*sub_y 
		output_ind = output<0
		gradient = lamb*w - np.divide(sum(np.multiply(sub_y[output_ind],sub_x[output_ind])),row_per_update) # row wise sum? divide by total A set? or total A wrong subset?

		lr = 1/(i*lamb)

		raw_w = w - lr*gradient
		w = np.minimum(1,1/np.sqrt(lamb)/np.sqrt(sum([i**2 for i in raw_w])))*raw_w

	return w


def adagrad(w, prediction, target, lr):
	"""
	lr = 1/sqrt(T)
	"""

	return






def PEGASOS_tf(x,y,T,lamb): #################################################### developing
	# tensorflow version of PEGASOS algorithm


	# initialize weights with normal of std = 0.01, mean = 0
	w = tf.variables(tf.random_normal([tf.shape(x)[1],1],stddev=0.01)/tf.sqrt(lamb), name='weights')

	row_per_update = tf.floor_div(tf.shape(x)[0], T)

	sub_x = tf.placeholder(tf.float32, shape=[row_per_update, tf.shape(x)[1]])
	sub_y = tf.placeholder(tf.float32, shape=[row_per_update, tf.shape(y)[1]])
	ite = tf.placeholder(tf.int32, shape=(1))


	output = tf.multiply(tf.matmul(sub_x,w),sub_y)
	output_ind = tf.greater(output,0) 
	gradient = tf.multiply(lamb,w) - tf.divide(tf.reduce_sum(\
				tf.multiply(tf.boolean_mask(sub_y,output_ind),tf.boolean_mask(sub_x,output_ind)), axis=0),row_per_update) 

	lr = 1/(ite*lamb)

	raw_w = w - lr*gradient
	update_w = w.assign(tf.minimum(1,1/tf.sqrt(lamb)/tf.norm(raw_w))*raw_w)

	# tf.sqrt(sum([i**2 for i in raw_w])) # is norm?

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
	    sess.run(init)

		for i in range(T):
			new_w = sess.run(update_w, feed_dict = {sub_x: x[i:(i+row_per_update),:],\
													sub_y: y[i:(i+row_per_update),:],\
													ite:   i})

	return new_w


# # initialize variables
# x = tf.variables()	
