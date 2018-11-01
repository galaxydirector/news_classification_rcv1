from optimizers import *
from data_import import *
import time
start = time.time()
sparse_data = import_data()
print(type(sparse_data))
train_x, train_y, test_x, test_y = data_set_sparse(sparse_data)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print("Datasets Imported. Time: "+str(int(time.time()-start))+" seconds.")
s = time.time()
# PEGASOS with minibatch
T = 10000
lamb = 0.01
w = PEGASOS(train_x,train_y,T,lamb,test_x,test_y)
print("PEGASOS Completed. Time: "+str(int(time.time()-s))+" seconds.")


