def get_accuracy(x,y,w):
	'''
	x: m x d data as csr_matrix
	y: m x 1 label as csr_matrix
	w: d x 1 weights as numpy array
	'''

	return round(int(sum(x.multiply(w.reshape(-1)).sum(1)>0))/x.shape[0],2)