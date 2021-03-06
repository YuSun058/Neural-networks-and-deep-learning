# From MNIST data set, we have 50000 data for training set and 10000, 10000 data for validation and test set, respectively.
# Layers: input layer, 1 hidden layer, output layer
# Neurons: specify the number of neurons in the three layers, respectively in a vector called "sizes"
#


# Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
from pytictoc import TicToc  # show the running time

# Construct the above neural network as follows


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialization from normal distributions. In the future, we'll use better initialization setups
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if 'a' is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, lr, test_data=None):
        """Train the neural network by mini-batch stochastic gradient descent method.  The 'training_data' is a list of tuples '(x,y)' representing the training inputs and desired outputs.  'epochs' specifies how many times we're going to go through the whole training_data in total.  'mini_batch_size' is the number of data used in each updation in SGD.  'lr' is learning rate.  If 'test_data' is provided, then the network will be evaluated against the test data after each epoch, and partial progress will be printed out.  This is useful for tracking progress, but it slows things down substantially."""
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            t = TicToc()
            t.tic()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                print("Epoch {} : {} / {}".format(j,
                                                  self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))
            t.toc()

    def update_mini_batch(self, mini_batch, lr):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The 'mini_batch' is a list of tuples '(x, y)', and 'lr' is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:  # send one training sample every time
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # after the for loop, 'nabla_b' and 'nabla_w' contain the parameters after sum_partial derivatives over the training samples in one mini-batch
        self.weights = [w-(lr/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(lr/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple '(nabla_b, nabla_w)' representing the gradient for the cost function C_x.  'nabla_b' and 'nabla_w' are layer-by-layer lists of numpy arrays, similar to 'self.biases' and 'self.weights'."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        # list to store all the activations, layer by layer and initialize it with the input x
        activations = [x]
        zs = []  # list to store all the z vectors, layer by layer and initialize it with an empty list
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])  # compute the delta for the output layer, note that '*' is component-wise while 'dot' is matrix multiplication
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.
        Here, since the cost function is half of MSE, then the partial derivatives should be output activations minus the benchmarks y."""
        return (output_activations-y)


# Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
