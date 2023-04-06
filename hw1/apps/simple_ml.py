import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open (image_filesname) as f:
      file_content = f.read()
    index = 0
    magic_number, image_num, row, col = struct.unpack_from ('>IIII', file_content, index)
    index += struct.calcsize ('>IIII')
    X_list = []
    for i in range(image_num):
      for j in range (row):
        for k in range (col):
          cur_byte = int(struct.unpack_from ('>B', file_content, index)[0])
          X_list.append (cur_byte)
          index += struct.calcsize ('>B')
    X = np.array(X_list, np.float32).reshape(image_num, row * col)
    min_value = np.min (X)
    max_value = np.max (X)
    X = (X - min_value) / (max_value - min_value)

    with gzip.open (label_filename) as f:
      file_content = f.read()
    index = 0
    magic_number, num = struct.unpack_from ('>II', file_content, index)
    index += struct.calcsize ('>II')
    Y_list = []
    for i in range(image_num):
      cur_byte = int(struct.unpack_from ('>B', file_content, index)[0])
      Y_list.append (cur_byte)
      index += struct.calcsize ('>B')
    y = np.array (Y_list, np.uint8)
    return (X, y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    Zy = ndl.multiply (Z, y_one_hot)
    loss = ndl.log (ndl.summation (ndl.exp (Z), axes=1)) - ndl.summation (Zy, axes=1)
    avg = ndl.divide_scalar (ndl.summation (loss), loss.shape[0])
    return avg
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples, input_dim = X.shape
    num_classes = W2.shape[1]
    for l in range (0, num_examples, batch):
      r = min (num_examples, l + batch)
      curX = ndl.Tensor (X[l:r][:], dtype="float32")
      Iy = np.zeros ((r - l, num_classes))
      Iy[np.arange (r - l), y[l:r]] = 1
      curY = ndl.Tensor (Iy, dtype="float32")
      loss = softmax_loss (ndl.matmul (ndl.relu (ndl.matmul (curX, W1)), W2), curY)
      loss.backward()
      W1grad = W1.grad
      W2grad = W2.grad
      W1 = W1.detach()
      W2 = W2.detach()
      W1 -= lr * W1grad
      W2 -= lr * W2grad
      W1 = W1.detach()
      W2 = W2.detach()
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
