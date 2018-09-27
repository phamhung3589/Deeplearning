import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', 'r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('datasets/test_signs.h5', 'r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    X_train = train_set_x_orig/255
    X_test = test_set_x_orig/255
    y_train = np.eye(6)[train_set_y_orig.reshape(-1)]
    y_test = np.eye(6)[test_set_y_orig.reshape(-1)]

    return X_train, y_train, X_test, y_test, classes


def create_placeholder(n_H, n_W, n_C, n_y):

    X = tf.placeholder(dtype=tf.float64, shape=[None, n_H, n_W, n_C])
    y = tf.placeholder(dtype=tf.float64, shape=[None, n_y])

    return X, y


def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", shape=[4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    W2 = tf.get_variable("W2", shape=[2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)

    parameters = {"W1": W1, "W2":W2}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    return Z3


def compute_cost(ZL, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=ZL))

    return cost


def random_minibatchs(X, y, minibatch_size):

    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffle_X = X[permutation, :, :, :]
    shuffle_Y = y[permutation, :]
    minibatchs = []
    end_minibatch = m//minibatch_size

    for i in range(end_minibatch):
        minibatch_X = shuffle_X[i*minibatch_size : (i+1)*minibatch_size, :, :, :]
        minibatch_Y = shuffle_Y[i*minibatch_size : (i+1)*minibatch_size, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    if m % minibatch_size != 0:
        minibatch_X = shuffle_X[end_minibatch*minibatch_size : , :, :, :]
        minibatch_Y = shuffle_Y[end_minibatch*minibatch_size : , :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatchs.append(minibatch)

    return minibatchs


def model(X_train, X_test, y_train, y_test, learning_rate=0.009, epoch_nums=100, minibatch_size=64):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    (m, n_H, n_W, n_C) = X_train.shape
    n_y = y_train.shape[1]
    costs = []

    # Create Placeholders of the correct shape
    X, y = create_placeholder(n_H, n_W, n_C, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ZL = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(ZL, y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        for epoch in range(epoch_nums):
            epoch_cost = 0
            minibatchs = random_minibatchs(X_train, y_train, minibatch_size)

            for minibatch in minibatchs:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost =  sess.run([optimizer, cost], feed_dict={X: minibatch_X, y:minibatch_Y})

                epoch_cost += minibatch_cost/len(minibatchs)

            if epoch % 5 == 0:
                print("Cost at iteration: {} {:.3f}".format(epoch, epoch_cost))
            if epoch % 1 == 0:
                costs.append(epoch_cost)

        # Calculate the correct predictions
        parameters = sess.run(parameters)
        predict_op = tf.argmax(ZL, axis=1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y, axis=1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, y: y_train})
        test_accuracy = accuracy.eval({X: X_test, y: y_test})
        print("train accuracy: ", train_accuracy)
        print("test accuracy: ", test_accuracy)

        # plot the cost
        plt.figure()
        plt.plot(costs)
        plt.xlabel("iterations per 5")
        plt.ylabel("cost")
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return train_accuracy, test_accuracy, parameters


def run_model():
    X_train, y_train, X_test, y_test, classes = load_dataset()
    train_accuracy, test_accuracy, parameters = model(X_train, X_test, y_train, y_test)
    fname = "datasets/my_image.jpg"
    image = np.array(ndimage.imread(fname, flatten=False), dtype=np.float64)
    my_image = scipy.misc.imresize(image, size=(64, 64))
    (n_H, n_W, n_C) = my_image.shape
    my_image = np.expand_dims(my_image, axis=0)
    X = tf.placeholder(dtype=tf.float64, shape=[1, n_H, n_W, n_C])
    ZL = forward_propagation(X, parameters)
    ZL = tf.argmax(ZL, axis=1)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print("label of this figure is: ", sess.run(ZL, feed_dict={X: my_image}))
    sess.close()
    plt.figure()
    plt.imshow(my_image[0])
    plt.show()


run_model()
