from os.path import dirname, join as pjoin
import numpy as np
import scipy.io as sio
import h5py

mat_fname = 'sample_data/mat/-8FLF-osZmA/00120.mat'
f = h5py.File(mat_fname, 'a')

# get all categories in the file
print("Keys: ")
print(list(f.items()))

# available catagories
#['#refs#', 'class', 'id', 'reBBox', 'reMask', 'reS_col', 'reS_id']

# choose a category to parse
dataset = f['id']
dataset2 = f['reBBox']
# get dataset shape
print("Shape: ")
print(dataset.shape)

# print dataset
print("Dataset: ")
print(dataset[0:])
print(dataset2[0:])

# If category is class
"""
# since it is a set of references, store reference
ref = dataset[0,0]

# print dataset
print("Dataset: ")
print(f[ref])
print(f[ref][0:10])
"""
