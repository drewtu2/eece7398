import numpy as np
import torch

# ## Construct a randomly initialized 5x3 matrix
# x = torch.rand(5, 3)        # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
# print(x)
#
# # Construct a matrix filled zeros and of dtype long:
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
# print(x.type())
# print(x.size())

#
# # Construct a tensor directly from data:
# tensor_1D = torch.tensor([1,2,3], dtype=torch.int)
# tensor_2D = torch.tensor( [[1,1], [2,2]], dtype=torch.double)
# print("tensor_1D: ", tensor_1D)
# print("tensor_2D: ", tensor_2D)



# # Create a tensor based on an existing tensor.
# These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
# x = torch.zeros(5, 3, dtype=torch.long)
# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(id(x))
# print(x)
# x = torch.randn_like(x, dtype=torch.float)    # will take x's size and override dtype, then filled with rand num
# print(x)
# print(id(x))
# result has the same size
# The default type is long or double depends on whether there is a '.' followed by the number

# # we can create different types of tensor such as long tensor, float tensor...
# data = [-1, 0, 1, 2]
# print(data)
# tensor = torch.tensor(data)
# tensor_float = torch.FloatTensor(data)
# print(tensor)
# print(tensor_float)
# # # torch.DoubleTensor(), torch.LongTensor(), torch.IntTensor()...


# # # convert tensor to/from numpy array
# tensor1 = torch.tensor([[1,2],[3,4]])
# print(type(tensor1))
#
# # # ---- to np array ----
# nparray = tensor1.numpy()
# print(nparray)
# print(type(nparray))
# #
# # # ---- np to tensor ----
# new_tensor = torch.from_numpy(nparray)
# print(new_tensor)
# print(new_tensor.type())


### multiply two tensors
### torch.mul, torch.mm, torch.matmul
# tensor_3x4 = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])      # 3x4 matrix
# tensor_1x4 = torch.tensor([1,2,3,4])                            # 1x4 matrix
# tensor_prod = torch.mul(tensor_3x4,tensor_1x4)                  # 3x4 matrix

# print(tensor_3x4)
# print(tensor_1x4)
# print("torch.mul:")
# print(tensor_prod)     # scalar multiplication, not matrix multiplication
# print("--------------")

# # ---- matrix multiplication ----
# tensor_a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])      # 3x3 matrix
# tensor_b = torch.tensor([1,2,3]).reshape(3,1)           # reshape 1x3 -> 3x1 matrix
# print(tensor_a)
# print(tensor_b)
# tensor_mm = torch.mm(tensor_a,tensor_b)                 # torch.mm() is used for calculating matrix multiplication
# print(tensor_mm)                                        # torch.mm() can only handle upto 2D matrix multiplication
#
# tensor_matmul = torch.matmul(tensor_a,tensor_b)         # torch.matmul() is also used for matrix multiplication
# print(tensor_matmul)                                    # but torch.matmul() can handle mat-mul with higher dimension


# ---- comparing cpu and gpu ----
# import time
# a = torch.rand(10000,10000)
# b = torch.rand(10000,10000)
# start = time.time()
# c = torch.mm(a,b)
# end = time.time()
# print("time using CPU: ", end-start)
# #
# # gpu = torch.device("cuda:0")
# # a = a.to(gpu)
# # b = b.to(gpu)
# a = a.cuda()      # same as .to(gpu)
# b = b.cuda()
# start = time.time()
# c = torch.mm(a,b)
# end = time.time()
# print("time using GPU: ", end-start)

#print(a.type())



# # shallow copy and deep copy
# import copy
# print("---- shallow copy ----")
# a = [1,2,3]
# b = a           # this is a shallow copy
# print("a:",a,"   b:",b)
# print(id(a))            # The id() function returns identity (unique integer) of an object
# print(id(b))
# print(id(a) == id(b))
#
# a[0] = 4
# print("a:",a)
# print("b:",b)
# #
# print("---- deep copy ----")
# c = [1,2,3]
# d = copy.deepcopy(c)       # deep copy
#
# print("c:",c,"   d:",d)
# print(id(c))            # The id() function returns identity (unique integer) of an object
# print(id(d))
# print(id(c) == id(d))
#
# c[0] = 666
# print("c:",c)
# print("d:",d)

# # during training process, watch weights between different steps

# or you can copy a tensor by using y = torch.Tensor(x)











