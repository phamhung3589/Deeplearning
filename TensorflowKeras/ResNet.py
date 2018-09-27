import numpy as np
import h5py
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, AveragePooling2D, Add
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
import keras.backend as K


def load_dataset():
    train_dataset = h5py.File("datasets/train_happy.h5")
    test_dataset = h5py.File("datasets/test_happy.h5")

    train_set_x_orig = np.array(train_dataset['train_set_x'])
    train_set_y_orig = np.array(train_dataset['train_set_y'])

    test_set_x_orig = np.array(test_dataset['test_set_x'])
    test_set_y_orig = np.array(test_dataset['test_set_y'])
    classes = np.array(test_dataset['list_classes'])

    train_set_y_orig = np.reshape(train_set_y_orig, (1, train_set_y_orig.shape[0])).T
    test_set_y_orig = np.reshape(test_set_y_orig, (1, test_set_y_orig.shape[0])).T

    X_train = train_set_x_orig/255
    X_test = test_set_x_orig/255

    y_train = np.eye(6)[train_set_y_orig.reshape(-1)]
    y_test = np.eye(6)[test_set_y_orig.reshape(-1)]

    return X_train, X_test, y_train, y_test, classes


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    # Retrieve Filters
    F1, F2, F3 = filters

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + "_branch"

    # First component of main path
    X = Conv2D(filters = F1, kernel_size=(1,1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + "_branch"

    # Save the input value
    X_shortcut = X
    # Retrieve Filters
    F1, F2, F3 = filters

    # First component of main path
    X = Conv2D(filters = F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size=(f, f), strides=(1, 1), padding='SAME', name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters = F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def resnet50_model(input_shape=(64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    # stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    #Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s = 1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')


    # stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s = 2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s = 2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s = 2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    # Create model
    model = Model(inputs=X_input, outputs = X, name="res_net_50")

    return model


def predict(model):
    img_path = "datasets/my_image.jpg"
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x, batch_size=64))
    # my_image = scipy.misc.imread(img_path)
    # plt.figure()
    # plt.imshow(my_image)
    # plt.show()

    return

def run_model():
    X_train, X_test, y_train, y_test, classes = load_dataset()
    model = resnet50_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(x=X_train, y=y_train, batch_size=64, epochs=2)
    score = model.evaluate(x=X_test, y=y_test)
    print("loss: = ", score[0])
    print("Accuracy = ", score[1])
    predict(model)

run_model()