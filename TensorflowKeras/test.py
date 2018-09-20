import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable("W" + str(l), shape=[layer_dims[l], layer_dims[l-1]],
                                                   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable("b" + str(l), shape=[layer_dims[l], 1], dtype=tf.float32, initializer=tf.zeros_initializer())

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
    permutation = list(np.random.permutation(L))
    shuffle_X = X[:, permutation]
    shuffle_Y = y[:, permutation]
    end_minibatch = L//mini_batch_size
    minibatchs = []

    for i in range(end_minibatch):
        minibatch_X = shuffle_X[:, i*mini_batch_size : (i+1)*mini_batch_size]
        minibatch_Y = shuffle_Y[:, i*mini_batch_size : (i+1)*mini_batch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    if L * mini_batch_size != 0:
        minibatch_X = shuffle_X[:, end_minibatch * mini_batch_size :]
        minibatch_Y = shuffle_Y[:, end_minibatch * mini_batch_size :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    return minibatchs


def model(X_train, y_train, X_test, y_test, layer_dims, learning_rate=0.0075, num_epochs=1500, mini_batch_size=64):
    (n_x, m) = X_train.shape
    n_y = y_train.shape[0]
    X, y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(layer_dims)

    ZL = forward_propagation(X, parameters)

    cost = compute_cost(ZL, y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            minibatchs = random_minibatchs(X_train, y_train, mini_batch_size)

            for minibatch in minibatchs:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, y: minibatch_Y})
                epoch_cost += cost / len(minibatchs)

            if epoch % 100 == 0:
                print("Cost after epoch", epoch, epoch_cost)

        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, y: y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, y: y_test}))
    return


def run_model():
    data = pd.read_csv('input.txt')
    data = shuffle(data)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    X_train = X_train.T.as_matrix()
    X_test = X_test.T.as_matrix()
    y_train = np.eye(5)[y_train.as_matrix().astype(int).reshape(-1)].T
    y_test = np.eye(5)[y_test.as_matrix().astype(int).reshape(-1)].T
    layer_dims = [X_train.shape[0], 6, 6, 6, 6, 6, y_train.shape[0]]
    parameters = model(X_train, y_train, X_test, y_test, layer_dims)
    return

run_model()




