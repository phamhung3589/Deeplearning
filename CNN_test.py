import numpy as np
import h5py
import matplotlib.pyplot as plt


def conv_single_step(a_prev_slice, W, b):
    """
    :param a_prev_slice: slice of input data of shape (f, f, n_C_prev)
    :param W: weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    :param b: bias parameters contained in a window - matrix of shape (1, 1, 1)
    :return: Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice and W. Do not add the bias here
    s = np.multiply(a_prev_slice, W)

    # Sum over all entries of the volumns s
    Z = np.sum(s)

    # Add the bias b t oZ. cast b to a float() so that Z results in a scalar value
    Z += float(b)

    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    pad = hparameters['pad']
    stride = hparameters['stride']

    # Compute the dimensions of the CONV output volumn using the formula given above.
    n_H = int((n_H_prev + 2*pad -f)/stride) + 1
    n_W = int((n_W_prev + 2*pad -f)/stride) + 1

    # Initialize the output volumn Z eith zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    # Create A_prev_pad by padding A_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    for i in range(m):                                  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                      # Select ith training example's padded activation
        for h in range(n_H):                            # loop over vertical axis of the output volume
            for w in range(n_W):                        # loop over horizontal axis of the output volume
                for c in range(n_C):                    # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell)
                    a_prev_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    Z[i,h,w,c] =  conv_single_step(a_prev_slice, W[:, :, :, c], b[:, :, :, c])

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def pool_forward(A_prev, hparameters, mode='max'):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']

    # Define the dimensions of the output
    n_H = int((n_H_prev -f)/stride) + 1
    n_W = int((n_W_prev -f)/stride) + 1
    n_C = n_C_prev
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):                              # loop over the training examples
        a_prev = A_prev[i]
        for h in range(n_H):                        # loop on the vertical axis of the output volume
            for w in range(n_W):                    # loop on the horizontal axis of the output volume
                for c in range(n_C):                # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev[vert_start:vert_end, horiz_start:horiz_end, c])

                    if mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev[vert_start:vert_end, horiz_start:horiz_end, c])

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    pad = hparameters['pad']
    stride = hparameters['stride']

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    dA_prev_pad = np.pad(dA_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    for i in range(m):                              # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):                        # loop over vertical axis of the output volume
            for w in range(n_W):                    # loop over horizontal axis of the output volume
                for c in range(n_C):                # loop over the channels of the output volume

                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db


def pool_backward(dA, cache, mode='max'):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    (A_prev, hparameters) = cache
    # (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape
    f = hparameters['f']
    stride = hparameters['stride']

    # Initialize dA_prev with zeros
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):                              # loop over the training examples
        # select training example from A_prev
        a_prev = A_prev[i]
        for h in range(n_H):                        # loop on the vertical axis
            for w in range(n_W):                    # loop on the horizontal axis
                for c in range(n_C):                # loop over the channels (depth)

                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f

                    # Compute the backward propagation in both modes.
                    if mode == 'max':
                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask*dA[i, h, w, c]

                    if mode == 'average':
                        # Compute dA_prev
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += (dA[i, h, w, c]/(f*f))*np.ones((f, f))

    return dA_prev


def run_conv_forward():
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
                   "stride": 2}

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

    np.random.seed(1)
    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))


def run_pool_forward():
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride" : 2, "f": 3}

    A, cache = pool_forward(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pool_forward(A_prev, hparameters, mode = "average")
    print("mode = average")
    print("A =", A)


def run_pool_backward():
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride" : 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])
    print()
    dA_prev = pool_backward(dA, cache, mode = "average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1,1])