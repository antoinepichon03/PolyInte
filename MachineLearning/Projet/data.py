import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

data = np.load('data/train.npz')
print(data.files) 

X = data['X_train']
y = data['y_train']
ids = data['ids']
X_sparse = sparse.csr_matrix(data['X_train'])
X_sparse_100000 = X_sparse[:100000]

