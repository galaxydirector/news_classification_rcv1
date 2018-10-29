# PEGASOS ALGO
# ADAGRAD ALGO
import numpy as np


def PEGASOS(x,y,T,lamb):





	# initialize a w with normal of std = 0.01
	w = 0.01*np.random.randn(x.shape[1],1)/np.sqrt(lamb) # how to set up lambda?

	for i in range(T):
		# what does a random subset means?
		# does that mean a element could be drawn more than once?
		# 1. if not allow to drawn more than once, then shuffle before enter 
		# 2. otherwise, np.random.randint(0,size) then take the elements

		sub_x =
		sub_y = 

		# note: numpy matmul is equivalent to dot
		# while numpy multipy is equivalent to *, which is element wise
		output = np.matmul(sub_x,w)*sub_y 
		output_ind = output<0
		gradient = lamb*w - sub_y[output_ind]*sub_x[output_ind] # row wise sum? divide by total A set? or total A wrong subset?

		lr = 1/(i*lamb)



	return

def adagrad(w, prediction, target, lr):
	"""
	lr = 1/sqrt(T)
	"""

	return
