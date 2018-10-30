# data import

from sklearn.datasets import fetch_rcv1
from scipy import sparse
import numpy as np

def import_data():
	rcv1 = fetch_rcv1()

	# do we have ways to simplify the way to find index of row contains 1?
	aa = rcv1['target'][:,33]
	kk = list(aa.toarray().reshape(-1,).astype("int"))
	postive_ind = [i for i, x in enumerate(kk) if x==1]
	negative_ind = [i for i, x in enumerate(kk) if x==0]

	# generate new -1 and 1 target
	# len(postive_ind)+len(negative_ind) = 804,414
	new_target = np.ones(804414) #check how long
	for i in negative_ind:
	    new_target[i] = -1

	new_rcv1 = sparse.hstack([rcv1['data'],new_target.reshape(-1,1)]) #804414x47237

	return new_rcv1

def data_set(sparse_data):
	arr_data = sparse_data.toarray()

	# split data
	training, test = arr_data[:100000,:], arr_data[100000:,:]

	training = numpy.random.shuffle(training)

	# split x and y
	train_x = training[:,:-1]
	train_y = training[:,-1]
	
	test_x = training[:,:-1]
	test_y = training[:,-1]

	return train_x, train_y, test_x, test_y



if __name__ == '__main__':
	main()

	