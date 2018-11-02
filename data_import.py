# data import

from sklearn.datasets import fetch_rcv1
from scipy import sparse
from scipy.sparse import csr_matrix
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
	csr_data = new_rcv1.tocsr()

	return csr_data


def data_set_sparse(sparse_data):
	'''
	reference: https://stackoverflow.com/questions/13843352/what-is-the-fastest-way-to-slice-a-scipy-sparse-matrix
	sparse_data: csr_matrix input
	return: train_x, train_y, test_x, test_y as csr_matrix, which supports slicing
	'''
	training, test = sparse_data[:100000,:], sparse_data[100000:,:]
	# split x and y
	train_x = training[:,:-1]
	train_y = training[:,-1]
	
	test_x = test[:,:-1]
	test_y = test[:,-1]

	return train_x, train_y, test_x, test_y

def dense_data_generator(x_data,y_data,T = None, one_hot = False):
	"""args: data must be csr form
	While T is None and one_hot is True, it returns line by line. Specifically for 
	neural nets training"""

	# shuffle the data
	# print(type(csr_data.get_shape()[0]))
	n_rows = x_data.get_shape()[0]
	shuffled_ind = np.array(range(n_rows))
	np.random.shuffle(shuffled_ind)
	shuffled_x = x_data[shuffled_ind,:]
	shuffled_y = y_data[shuffled_ind,:]

	if T is not None:
		row_per_update = n_rows // T
		batch = [row_per_update*i for i in range(T)]
	else: 
		row_per_update = 1
		batch = range(n_rows)
	
	for i in batch:
		#print(i)
		sub_x = shuffled_x[i:(i+row_per_update),:]
		sub_y = shuffled_y[i:(i+row_per_update),:]

		if T is None and one_hot is True:
			print(sub_y.toarray().reshape(-1)[0])
			if sub_y.toarray().reshape(-1)[0] == -1:
				print(np.array([1,0]).shape)
				yield (sub_x.toarray(), np.array([1,0]))
				# onehot_y = np.array([1,0])
			elif sub_y.toarray().reshape(-1)[0] == 1:
				print(np.array([0,1]).shape)
				yield (sub_x.toarray(), np.array([0,1]))

			# 	onehot_y = np.array([0,1])
			# print("shape",onehot_y.shape)
			# yield (sub_x.toarray(),onehot_y)


		else:
			yield (sub_x.toarray(),sub_y.toarray())



if __name__ == '__main__':
	csr_data = import_data()
	kkk = dense_data_generator(csr_data, 10000)
	fucker = next(kkk)
	print(fucker[0])