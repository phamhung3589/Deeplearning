import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float64, shape=[n_x, None])
    Y = tf.placeholder(tf.float64, shape=[n_y, None])

    return X, Y


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable("W" + str(l), shape=[layer_dims[l], layer_dims[l-1]],
                                                   dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['b' + str(l)] = tf.get_variable("b" + str(l), shape=[layer_dims[l], 1], dtype=tf.float64, initializer=tf.zeros_initializer())

    return parameters


def forward_propagation(X, parameters):
    L = len(parameters)//2
    A = X
    for l in range(1, L):
        Z = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])
        A = tf.nn.relu(Z)

    Z = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])

    return Z


def compute_cost(ZL, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.transpose(y), logits=tf.transpose(ZL)))

    return cost


def random_minibatchs(X, y, mini_batch_size):
    L = X.shape[1]
    permutation = np.random.permutation(L)
    shuffle_X = X[:, permutation]
    shuffle_Y = y[:, permutation]
    end_minibatch = L//mini_batch_size
    minibatchs = []

    for i in range(end_minibatch):
        minibatch_X = shuffle_X[:, i*mini_batch_size : (i+1)*mini_batch_size]
        minibatch_Y = shuffle_Y[:, i*mini_batch_size : (i+1)*mini_batch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    if L % mini_batch_size != 0:
        minibatch_X = shuffle_X[:, end_minibatch * mini_batch_size :]
        minibatch_Y = shuffle_Y[:, end_minibatch * mini_batch_size :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    return minibatchs


def model(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.0075, num_epochs=1000, mini_batch_size=64, print_cost=True):
    (n_x, m) = X_train.shape
    n_y = y_train.shape[0]
    costs = []

    # Create place holder for X and y with shape: X: [n_x, None], Y: [n_y, None]
    X, y = create_placeholders(n_x, n_y)

    # initial parameters for neural network
    parameters = initialize_parameters(layer_dims)

    # compute forward propagation with return value is Z for the output layer
    ZL = forward_propagation(X, parameters)

    # compute cost with ZL and y
    cost = compute_cost(ZL, y)

    # initial back propagation is adam function
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # create global variable
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0

            # Create mini batch from X_train and y_train
            minibatchs = random_minibatchs(X_train, y_train, mini_batch_size)

            # Loop over minibatch
            for minibatch in minibatchs:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, y: minibatch_Y})
                epoch_cost += minibatch_cost / len(minibatchs)

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch", epoch, epoch_cost)
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.figure()
        plt.plot(costs)
        plt.xlabel("Number of iteration mul 5")
        plt.ylabel("cost of algorithm")
        # plt.show()

        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype="float"))
        print("accuracy of training data: ", accuracy.eval({X: X_train, y: y_train}))
        print("Accuracy of test data: ", accuracy.eval({X: X_test, y:y_test}))

    return parameters


def plot_graph(X, y, parameters):
    y_true = np.argmax(y, axis=0)
    ZL = forward_propagation(X, parameters)
    sess = tf.Session()
    y_hat = sess.run(tf.argmax(ZL))

    plt.figure()
    plt.plot(X[0, y_true == 0], X[1, y_true == 0], "c*", markersize=5, label="Class 1")
    plt.plot(X[0, y_true == 1], X[1, y_true == 1], "bx", markersize=5, label="Class 2")
    plt.plot(X[0, y_true == 2], X[1, y_true == 2], "r+", markersize=5, label="Class 3")
    plt.plot(X[0, y_true == 3], X[1, y_true == 3], "g^", markersize=5, label="Class 4")
    plt.plot(X[0, y_true == 4], X[1, y_true == 4], "y>", markersize=5, label="Class 5")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("true graph")
    plt.legend()

    plt.figure()
    plt.plot(X[0, y_hat == 0], X[1, y_hat == 0], "c*", markersize=5, label="Class 1")
    plt.plot(X[0, y_hat == 1], X[1, y_hat == 1], "bx", markersize=5, label="Class 2")
    plt.plot(X[0, y_hat == 2], X[1, y_hat == 2], "r+", markersize=5, label="Class 3")
    plt.plot(X[0, y_hat == 3], X[1, y_hat == 3], "g^", markersize=5, label="Class 4")
    plt.plot(X[0, y_hat == 4], X[1, y_hat == 4], "y>", markersize=5, label="Class 5")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("prediction graph")
    plt.legend()

    # Plot decision boundary
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = tf.argmax(forward_propagation(np.c_[xx.ravel(), yy.ravel()].T, parameters))
    Z = tf.reshape(Z, shape=xx.shape)
    Z = sess.run(Z)
    sess.close()
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y_true, cmap=plt.cm.Spectral)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("prediction boundary")

    plt.show()


def run_model():
    data = pd.read_csv('input.txt')
    data = shuffle(data)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train.T.as_matrix()
    X_test = X_test.T.as_matrix()
    y_train = np.eye(5)[y_train.as_matrix().astype(int).reshape(-1)].T
    y_test = np.eye(5)[y_test.as_matrix().astype(int).reshape(-1)].T
    layer_dims = [X_train.shape[0], 6, 6, 6, 6, 6, y_train.shape[0]]
    parameters = model(X_train, y_train, X_test, y_test, layer_dims)
    plot_graph(X_train, y_train, parameters)

run_model()






