import tensorflow as tf

# 0D tensor
d0 = tf.constant(3)
print("\n0D tensor:")
print(d0)

# 1D tensor
d1 = tf.constant([1, 2, 3, 4, 5])
print("\n1D tensor:")
print(d1)

# 2D tensor
d2 = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print("\n2D tensor:")
print(d2)

# float 2D tensor
d2f = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=tf.float32)
print("\nfloat 2D tensor:")
print(d2f)

# 3D tensor
d3 = tf.constant([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
print("\n3D tensor:")
print(d3)


import numpy as np

nparr = np.array([1, 2, 3, 4, 5])
print("\nnparr:")
print(nparr)

# convert np array to tensor
tensor = tf.convert_to_tensor(nparr, dtype=tf.float64)
print("\ntensor:")


# eye tensor
# batch_shape is the number of matrices
eye = tf.eye(
    num_rows=3,
    num_columns=3,
    batch_shape=(2, ), # add a dimension to the tensor
    dtype=tf.float32,
    name=None
)
print("\neye:")
print(eye)

# fill tensor # create a tensor with a specific value in all cells
fill = tf.fill(
    dims=(2, 3), # shape of the tensor
    value=9,
)

# ones tensor
ones = tf.ones(
    shape=(2, 3),
    dtype=tf.float32,
    name=None
)

# ones_like tensor
ones_like = tf.ones_like(
    input=ones,
    dtype=None,
    name=None,
)

# shape tensor # shape is the number of elements in each dimension
shape = tf.shape(
    input=ones,
    name=None,
    out_type=tf.dtypes.int32,
)


# rank tensor  # rank is the number of dimensions
rank = tf.rank(
    input=ones,
    name=None,
)

# size tensor # size is the number of elements in the tensor
size = tf.size(
    input=ones,
    name=None,
    out_type=tf.dtypes.int32,
)

# random tensor # random is a tensor with random values
random = tf.random.normal(
    shape=(2, 3),
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None,
)

# random uniform tensor # random uniform is a tensor with random values between 
# minval and maxval
random_uniform = tf.random.uniform(
    shape=(2,),
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None,
)

#argmax tensor # argmax is the index of the maximum value in the tensor
argmax = tf.argmax(
    input=random_uniform,
    axis=0, #axis is the dimension to reduce. If None, reduce all dimensions
    #out put is position of the maximum value in the tensor
    output_type=tf.dtypes.int64,
    name=None,
)
#argmin tensor # argmin is the index of the minimum value in the tensor

#pow tensor # pow is the power of the tensor
pow = tf.pow(
    x=random_uniform,
    y=tf.constant(2, dtype=tf.float32),
    name=None,
)

#reduce_sum tensor # reduce_sum is the sum of the tensor
reduce_sum = tf.reduce_sum(
    input_tensor=random_uniform,
    axis=None,
    keepdims=False,
    name=None,
)

#reduce_max tensor # reduce_max is the maximum value in the tensor
reduce_max = tf.reduce_max(
    input_tensor=random_uniform,
    axis=None,
    keepdims=False,
    name=None,
)

#top_k tensor # top_k is the k largest values in the tensor
#k is the number of largest values to find
top_k = tf.math.top_k(
    input=random_uniform,
    k=2,
    sorted=True,
    name=None,
)