#%% packages
import torch
import numpy as np

#%% list to tensor
my_list = [1, 2, 3, 4, 5]
my_tensor1 = torch.tensor(my_list)
my_tensor1

#%% numpy array to tensor
my_data = np.array([[1, 2, 3], [4, 5, 6]])
my_tensor2 = torch.from_numpy(my_data)
my_tensor2

#%% tensor from zeros
shape = (3, 2)
my_tensor3 = torch.zeros(shape)
my_tensor3
#%% tensor from random values
my_tensor4 = torch.rand(shape)
my_tensor4


#%% attributes of a tensor
print(f"Shape of my_tensor4: {my_tensor4.shape}")
print(f"Type of my_tensor4: {my_tensor4.dtype}")
print(f"Device of my_tensor4: {my_tensor4.device}")
print(f"Gradient of my_tensor4: {my_tensor4.requires_grad}")

#%% set gradient to true
my_tensor4.requires_grad = True
print(f"Gradient of my_tensor4: {my_tensor4.requires_grad}")

#%% indexing and slicing
print("first row: ", my_tensor4[0])
print("first column: ", my_tensor4[:, 0])
print("last column: ", my_tensor4[:, -1])


# %%
