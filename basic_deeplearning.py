import numpy as np
import pandas as pd
from scipy.sparse.csgraph._traversal import depth_first_order
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

def initialize_parameters(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def softmax(Z):
    e_z = np.exp(Z)
    A = (e_z / np.sum(e_z, axis=0))
    return A, Z

def sigmoid(Z):
    A = 1.0/(1 + np.exp(-Z))
    return A, Z

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def dropout_forward(A, keep_prob):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < keep_prob
    A = np.multiply(A, D)/keep_prob
    return A, D

def linear_activation_forward(A_prev, W, b, activation, keep_prob):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        if keep_prob != None:
            A, dropout_cache = dropout_forward(A, keep_prob)
        else:
            dropout_cache = 0

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
        if keep_prob != None:
            A, dropout_cache = dropout_forward(A, keep_prob)
        else:
            dropout_cache = 0
    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        if keep_prob != None:
            A, dropout_cache = dropout_forward(A, keep_prob)
        else:
            dropout_cache = 0

    cache = linear_cache, activation_cache, dropout_cache

    return A, cache

def forward_propagation(X, parameters, keep_prob):
    L = len(parameters)//2
    caches = []
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu", keep_prob)
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax", None)
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1.0/m * np.sum((np.multiply(np.log(AL),Y)) + np.multiply(np.log(1-AL), (1-Y)))
    return cost

def compute_cost_with_regularization(AL, Y, lambd, parameters):
    m = Y.shape[1]
    cost = -1.0 / m * np.sum((np.multiply(np.log(AL), Y)) + np.multiply(np.log(1 - AL), (1 - Y)))
    cost_regularization = lambd/(2*m)* np.sum([np.sum(np.square(value)) for key, value in parameters.items()  if "W" in key])
    return cost - cost_regularization

def linear_backward_with_regularization(dZ, cache, lambd):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1.0/m * np.dot(dZ, A_prev.T) + lambd/m*W
    db = 1.0/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1.0/(1 + np.exp(-Z))
    dZ = dA*s*(1-s)
    return dZ

def dropout_backward(dA, dropout_cache, keep_prob):
    dA = np.multiply(dA, dropout_cache)/keep_prob
    return dA

def linear_activation_backward_with_regularization(dA, cache, activation, lambd, keep_prob=None):
    linear_cache, activation_cache, dropout_cache = cache
    if keep_prob != None:
        dA = dropout_backward(dA, dropout_cache, keep_prob)

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_with_regularization(dZ, linear_cache, lambd)
    else:
        dZ = dA
        dA_prev, dW, db = linear_backward_with_regularization(dZ, linear_cache, lambd)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches, lambd, keep_prob):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = AL - Y
    current_cache = caches[L-1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward_with_regularization(dAL, current_cache, "softmax", lambd)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward_with_regularization(grads["dA" + str(l + 2)], current_cache,"relu", lambd, keep_prob)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate*grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate*grads['db' + str(l+1)]

    return parameters

def model(X, Y, layer_dims, learning_rate=0.05, num_iteration=5000, lambd=0.0001, keep_prob=1):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(num_iteration):
        AL, caches = forward_propagation(X, parameters, keep_prob)
        # cost = compute_cost(AL, Y)
        cost = compute_cost_with_regularization(AL, Y, lambd, parameters)
        grads = backward_propagation(AL, Y, caches, lambd, keep_prob)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print(i, cost)
    yhat = np.array([int(AL.T[i, j] == np.max(AL.T[i])) for i in range(AL.T.shape[0]) for j in range(AL.T.shape[1])])
    yhat = yhat.reshape(AL.T.shape).T
    accuracy = np.count_nonzero((yhat + Y) == 2) / Y.shape[1]
    print("Accuracy of algorithm: ", accuracy)
    return parameters

def run_model():
    data = pd.read_csv('input.txt')
    layer_dims = [2,6,6,6,6,6,5]
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lb = preprocessing.LabelBinarizer()
    X_train = X_train.T.as_matrix()
    X_test = X_test.T.as_matrix()
    y_train = y_train.T.as_matrix()
    y_train = lb.fit_transform(y_train).T
    y_test = y_test.T.as_matrix()
    y_test = lb.transform(y_test).T
    model(X_train, y_train, layer_dims)
    return 0

run_model()
