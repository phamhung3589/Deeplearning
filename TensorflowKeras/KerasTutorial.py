import numpy as np
import h5py
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import layer_utils, plot_model
from matplotlib.pyplot import imshow

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

    return X_train, X_test, train_set_y_orig, test_set_y_orig, classes


def happy_model(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(shape=input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D(padding=(3,3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(filters=32, kernel_size=(7,7),strides=(1,1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2,2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(units=1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='happy_model')

    return model


def predict(model):
    img_path = "datasets/happy_house.jpg"
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model.predict(x, batch_size=64)

    return


def run_model():
    X_train, X_test, y_train, y_test, classes = load_dataset()

    # Create the model
    model = happy_model(X_train.shape[1:])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on train data
    model.fit(x=X_train, y=y_train, epochs=10, batch_size=64)

    # Test the model on test data
    preds = model.evaluate(X_test, y_test)

    print("loss = ", preds[0])
    print("accuracy = ", preds[1])
    print("prediction: ", model.predict(x=X_test, batch_size=64))
    print(model.summary())

    # plot_model(happy_model, to_file="happy_model.png")
    # SVG(model_to_dot(happy_model).create(prog='dot', format='svg'))

    return preds

run_model()