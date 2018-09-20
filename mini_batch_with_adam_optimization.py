import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def initialize_parameters(layer_dims):
    """
    :param layer_dims: dimension of all hidden layer in neural network
    :return: parameters initialized randomly with "He" method.
    """
    parameters = {}
    v = {}
    s = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    # initialize adam parameter
    for l in range(len(parameters)//2):
        v['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        v['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
        s['dW' + str(l+1)] = np.zeros(parameters['W' + str(l+1)].shape)
        s['db' + str(l+1)] = np.zeros(parameters['b' + str(l+1)].shape)
    return parameters, v, s


def linear_forward(A, W, b):
    """
    :param A:     A of previous layer
    :param W, b:  parameters of current layer
    :return:      value Z of current layer and save cache(A, W, b) for back propagation
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def sigmoid(Z):
    A = 1.0/(1 + np.exp(-Z))

    return A, Z


def softmax(Z):
    Z = np.array(Z, dtype=np.float64)
    e_z = np.exp(Z)
    A = e_z / np.sum(e_z, axis=0)

    return A, Z


def relu(Z):
    A = np.maximum(0, Z)

    return A, Z


def tanh(Z):
    e_z = np.exp(2*Z)
    A = (e_z - 1)/(e_z + 1)

    return A, Z


def linear_activation_forward(A_prev, W, b, activation):
    """
    :param activation: assign activation function for each layer
    :return: with A_prev, W, b => compute Z and A of current layer
    """
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    else:
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    caches = linear_cache, activation_cache

    return A, caches


def forward_propagation(X, parameters):
    """
    :return: compute forward propagation (A, Z) for all hidden layer and output layer, assign activation function
             "relu" for hidden layer and "softmax" function for output layer
    """
    L = len(parameters)//2      # Length of all layer except input layer
    caches = []                 # save cache for each layer to compute back propagation
    A = X
    for l in range(1, L):
        A_prev = A
        A, current_cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(current_cache)
    AL, current_cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'softmax')
    caches.append(current_cache)

    return AL, caches


def compute_cost(AL, y, parameters, lambd):
    m = y.shape[1]
    cost = -1./m * np.sum(np.multiply(y, np.log(AL + 1e-10)) + np.multiply((1 - y), np.log(1 - AL + 1e-10)))
    cost_regularization = lambd/(2.*m)*np.sum([np.sum(np.square(value)) for key, value in parameters.items() if "W" in key])

    return cost - cost_regularization


def linear_backward(dZ, cache, lambd):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1.0/m*(np.dot(dZ, A_prev.T)) + lambd/m*W
    db = 1.0/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    Z = cache
    e_z = 1.0 / (1 + np.exp(-Z))
    dZ = dA * e_z * (1 - e_z)

    return dZ


def softmax_backward(dA, cache):
    Z = cache
    e_z = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    dZ = dA * e_z * (1 - e_z)

    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = dA * np.int64(Z > 0)
    dZ = np.nan_to_num(dZ + 1e-10)
    return dZ


def tanh_backward(dA, cache):
    Z = cache
    e_z = (np.exp(2*Z) - 1) / (np.exp(2*Z) + 1)
    dZ = dA * (1 - e_z) * (1 + e_z)

    return dZ


def linear_activation_backward(dA, cache, lambd, activation):
    """
    :param dA:              dA at current layer
    :param cache:           include linear activation (A_prev, W, b) and activation cache (Z)
    :param lambd:           parameter of regularization
    :param activation:      Used to compute derivative at softmax or relu layer
    :return:                from dA compute dZ at current layer, using dZ and linear cache to compute dA_prev, dW, db
    """
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    else:
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def backward_propagation(AL, y, caches, lambd):
    L = len(caches)
    grads = {}
    current_cache = caches[L-1]
    dAL = - (np.divide(y, AL+1e-10)) + np.divide((1 - y), (1 - AL+1e-10))
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, lambd, 'softmax')
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads['dA' + str(l+2)], current_cache, lambd, 'relu')
        grads['dA' + str(l+1)] = dA_prev
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate, v, s, t, beta1 = 0.95, beta2 = 0.9999, epsilon = 1e-8):
    """
    :param parameters:          parameters need to update
    :param grads:               dictionary of derivative of parameter in all layer
    :param learning_rate:       using this parameter to update
    :return:                    updated parameters with formula: W = W - learning_rate * dW
    """
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v['dW' + str(l+1)] = beta1*v['dW' + str(l+1)] + (1 - beta1)*grads['dW' + str(l+1)]
        v['db' + str(l+1)] = beta1*v['db' + str(l+1)] + (1 - beta1)*grads['db' + str(l+1)]

        v_corrected['dW' + str(l+1)] = v['dW' + str(l+1)]/(1 - np.power(beta1, t))
        v_corrected['db' + str(l+1)] = v['db' + str(l+1)]/(1 - np.power(beta1, t))

        s['dW' + str(l+1)] = beta2*s['dW' + str(l+1)] + (1 - beta2)*np.square(grads['dW' + str(l+1)])
        s['db' + str(l+1)] = beta2*s['db' + str(l+1)] + (1 - beta2)*np.square(grads['db' + str(l+1)])

        s_corrected['dW' + str(l+1)] = s['dW' + str(l+1)]/(1 - np.power(beta2, t))
        s_corrected['db' + str(l+1)] = s['db' + str(l+1)]/(1 - np.power(beta2, t))

        parameters['W' + str(l+1)] -= learning_rate*v_corrected['dW' + str(l+1)]/np.sqrt(s_corrected['dW' + str(l+1)] + epsilon)
        parameters['b' + str(l+1)] -= learning_rate*v_corrected['db' + str(l+1)]/np.sqrt(s_corrected['db' + str(l+1)] + epsilon)

    return parameters, v, s


def mini_batch(X, y, mini_batch_size):
    """
    :param mini_batch_size: size for mini batch (2^k: 16, 32, 64, 128, 256, 512)
    :return:
    """
    L = X.shape[1]
    minibatchs = []
    permutation = list(np.random.permutation(L))
    shuffle_X = X[:, permutation]
    shuffle_Y = y[:, permutation]
    end_minibatch = L//mini_batch_size
    for i in range(end_minibatch):
        minibatch_X = shuffle_X[:, i*mini_batch_size:(i+1)*mini_batch_size]
        minibatch_Y = shuffle_Y[:, i*mini_batch_size:(i+1)*mini_batch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    if L%mini_batch_size != 0:
        minibatch_X = shuffle_X[:, end_minibatch*mini_batch_size:]
        minibatch_Y = shuffle_Y[:, end_minibatch*mini_batch_size:]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    return minibatchs


def run_model(X, y, epoch, learning_rate, lambd, layer_dims, mini_batch_size, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    :param X:                   Input data X: dimension (n_x, m_example)
    :param y:                   label of data X
    :param epoch:               Number of iteration in neural network
    :param learning_rate:       Using in update parameters
    :param lambd:               parameters of regularization
    :param layer_dims:          list of layer in neural network - using in initial parameters
    :param mini_batch_size:     size of mini batch
    :return:                    parameters after training
    """
    parameters, v, s = initialize_parameters(layer_dims)
    costs = []
    t = 0
    # Loop over epoch to update all neural network
    for i in range(epoch):
        epoch_cost = 0

        # Get mini batch with in pute X
        minibatchs = mini_batch(X, y, mini_batch_size)

        # Loop over number of minibatch data to fast converge.
        for j in range(len(minibatchs)):
            minibatch_X, minibatch_Y = minibatchs[j]
            AL, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)
            grads = backward_propagation(AL, minibatch_Y, caches, lambd)
            t = t+ 1
            parameters, v, s = update_parameters(parameters, grads, learning_rate, v, s, t)

            # compute cost for epoch i: epoch_cost = (cost[0] + cost[1] + ... + cost[len(minibatchs)])/len(minibatchs)
            epoch_cost += cost/len(minibatchs)

        if i % 100 == 0:
            costs.append(epoch_cost)
            print("cost at iteration: {}: {:.2f}".format(i, epoch_cost))

    return parameters


def predict(X, y, parameters):
    """
    :param parameters: parameters after training in neural network.
    :return: Using tuned parameters to predict label of data X => use function "forward propagation"
    """
    AL, _ = forward_propagation(X, parameters)
    yhat = np.argmax(AL, axis=0)
    ytrue = np.argmax(y, axis=0)
    accuracy = np.count_nonzero((yhat - ytrue) == 0) / y.shape[1]

    return accuracy, (yhat, ytrue)


def predict_decision_boundary(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predict = np.argmax(AL, axis=0)

    return predict


def plot_graph(X, y_predict, y_true):
    """
    :return: plot graph of dimension (1, 2) of X with 2 types of label: y_predict and y_true
    """

    # plot predict graph of predict label
    plt.figure()
    plt.plot(X[0, y_predict == 0], X[1, y_predict == 0], 'c*', markersize=5, label = "class 1")
    plt.plot(X[0, y_predict == 1], X[1, y_predict == 1], 'b>', markersize=5, label = "class 2")
    plt.plot(X[0, y_predict == 2], X[1, y_predict == 2], 'y^', markersize=5, label = "class 3")
    plt.plot(X[0, y_predict == 3], X[1, y_predict == 3], 'g<', markersize=5, label = "class 4")
    plt.plot(X[0, y_predict == 4], X[1, y_predict == 4], 'rx', markersize=5, label = "class 5")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Predict graph")
    plt.legend()

    # plot true graph of true label
    plt.figure()
    plt.plot(X[0, y_true == 0], X[1, y_true == 0], 'c*', markersize=5, label = "class 1")
    plt.plot(X[0, y_true == 1], X[1, y_true == 1], 'b>', markersize=5, label = "class 2")
    plt.plot(X[0, y_true == 2], X[1, y_true == 2], 'y^', markersize=5, label = "class 3")
    plt.plot(X[0, y_true == 3], X[1, y_true == 3], 'g<', markersize=5, label = "class 4")
    plt.plot(X[0, y_true == 4], X[1, y_true == 4], 'rx', markersize=5, label = "class 5")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("True graph")
    plt.legend()



def plot_decision_boundary(model, X, y):
    """
    :param model: predict function of grid value z.
    :param X: training data X
    :param y: label of training data
    :return: graph of decision boundary
    """

    # find min and max of X[0] and X[1]. to plot on 2D flat
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.01

    # Create meshgrid of value from xmin to xmax of 2 Dimension of X[0] and X[1]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour of training example
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

    # plot training point of traning data X
    plt.scatter(X[0, :], X[1, :], c=np.argmax(y, axis=0), cmap=plt.cm.Spectral)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


data = pd.read_csv('input.txt')
data = shuffle(data)
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train = X_train.T.as_matrix()
X_test = X_test.T.as_matrix()
y_train = np.eye(5)[y_train.as_matrix().astype(int).reshape(-1)].T
y_test = np.eye(5)[y_test.as_matrix().astype(int).reshape(-1)].T
layer_dims = [2, 6, 6, 6, 6, 6, 5]

# training model
parameters = run_model(X_train, y_train, 1500, 0.0075, 0.0001, layer_dims, 128)

# predict and plot graph
score_train, value = predict(X_train, y_train, parameters)
yhat, ytrue = value
score_test, _ = predict(X_test, y_test, parameters)
print("train: ", score_train, "score test: ", score_test)
plot_graph(X_train, yhat, ytrue)
plot_decision_boundary(lambda x: predict_decision_boundary(x.T, parameters), X_train, y_train)