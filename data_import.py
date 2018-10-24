# data import

from sklearn.datasets import fetch_rcv1
from scipy import sparse,tocsr
import numpy as np
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