import numpy as np
import random


def PEGASOS(x,y,T,lamb,k):
	'''
	reference: https://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf
	x: m x d data
	y: m x 1 label
	T: max iterations
	lamb: chosen from 0.0001, 0.001, 0.01, 0.1, 1, 10, 100
	k: batch size
	'''
	m = x.shape[0]
	# initialize a w with normal of std = 0.01
	w = 0.01*np.random.randn(x.shape[1],1)/np.sqrt(lamb) # start with very small numbers with mean of 0

	for i in range(1,T+1):
		batch = [random.randint(0,m-1) for i in range(k)]
		sub_x = x[batch,:]
		sub_y = y[batch,:]
		# note: numpy matmul is equivalent to dot
		# while numpy multipy is equivalent to *, which is element wise
		output = np.matmul(sub_x,w)*sub_y 
		# should here be 0 or 1?
		output_ind = output<1
		gradient = lamb*w - np.divide(sum(np.multiply(sub_y[output_ind],sub_x[output_ind])),k)
		lr = 1/(i*lamb)
		raw_w = w - lr*gradient
		w = np.minimum(1,1/np.sqrt(lamb)/np.sqrt(sum([i**2 for i in raw_w])))*raw_w

	return w