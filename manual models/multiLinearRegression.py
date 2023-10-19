import copy
import math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

def predict_single_loop(x, w, b):
    """
    single predict using linear regression

    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     

    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p


def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 

    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    # cost formula: J(w,b) = 1/2m (sum (f_wb(x_i) - y_i)^2)
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    # gradient formula: dj_dw = 1/m (sum (f_wb(x_i) - y_i) * x_i)
    #                   dj_db = 1/m (sum (f_wb(x_i) - y_i))
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    # divide by m is done because we are taking the average of the gradients of all the examples

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    # MSE loss function: L(w) = (1/2m) * sum_i=1^m (f_wb(x_i) - y_i)^2

    # gradient descent formula: w = w - alpha * dj_dw
    #                           b = b - alpha * dj_db
    #                           J = cost_function(X, y, w, b)
    # dj = derivative of cost function w.r.t. w or b

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)  # None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  # None
        b = b - alpha * dj_db  # None

        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# get a row from our training data
x_vec = X_train[0, :]

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)

# get a row from our training data
x_vec = X_train[0, :]

# make a prediction
f_wb = predict(x_vec, w_init, b_init)

# Compute and display cost using our pre-chosen optimal parameters.
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}\n')

# Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}\n')

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m, _ = X_train.shape
for i in range(m):
    print(f"prediction: {predict(X_train[i], w_final, b_final)}, target value: {y_train[i]}")

print ("\n\n\n")












# normalize data
X_norm, _, _= zscore_normalize_features(X_train)

# some gradient descent settings
iterations = 1000
alpha = 1.0e-1

# run gradient descent
w_norm, b_norm, J_hist_norm = gradient_descent(X_norm, y_train, initial_w, initial_b, 
                                                compute_cost, compute_gradient,
                                                alpha, iterations)

print(f"b,w found by gradient descent: {b_norm:0.2f},{w_norm} ")

m, _ = X_norm.shape
for i in range(m):
    # print (X_norm[i])
    # print (predict(X_norm[i], w_norm, b_norm))
    print(f"prediction: {predict(X_norm[i], w_norm, b_norm)}, target value: {y_train[i]}")



# fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X[:,i],y)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("y")
# plt.show()






# x = np.arange(0,20,1)
# y = np.cos(x/2)



# X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# X = np.array(X)


# model_w,model_b, _ = gradient_descent(X, y, np.zeros(13), 0, compute_cost, compute_gradient, 0.1, 1000000)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
# plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()