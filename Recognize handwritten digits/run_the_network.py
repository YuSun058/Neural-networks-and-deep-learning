# Import the data & the neural network
import mnist_loader
import my_network
from pytictoc import TicToc

t = TicToc()
t.tic()

# Split the data into different sets
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Construct the neural network with 784 neurons in the input layer, 50 neurons in the
# hidden layer, and 10 neurons in the output layer
net = my_network.Network([784, 50, 10])

# Strat training and testing meanwhile
# Setting: 30 epochs, mini-batch size:10, learning rate: 2.5
net.SGD(training_data, 30, 10, 1.5, test_data=test_data)

t.toc()
