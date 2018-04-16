# This free software is licensed under GPL version 3 without any implied or
# explicit warranties. This copyright notice must be included in all
# non-machine-executable copies or derivative works of this software.
from FFNN import FFNN
import numpy as np

# Inputs and expected outputs
x = []
y = []
x_test = []
y_test = []

def dummy_function(x):
    return [x*np.sin(2*np.pi/(x+1))/10, x*np.cos(2*np.pi/(2*x+1))/1000]

# Training set
for i in range(0, 50):
    x.append([i])
    y.append(dummy_function(i))

# Testing set
for i in range(125, 200):
    x_test.append([i])
    y_test.append(dummy_function(i))

x = np.asarray(x).T
y = np.asarray(y).T
x_test = np.asarray(x_test).T
y_test = np.asarray(y_test).T

###################################################################################################
# 2D plot that tests different learning rates
###################################################################################################
import matplotlib.pyplot as plt

number_of_epochs = 2000
network_structure = [1, 4, 10, 2]
current_batch_size = 128
testing_frequency = 100
current_learning_rate = 0.00003
learning_rate_increment = 0.00005

# For plotting 
epochs = np.arange(0, number_of_epochs, step=testing_frequency)
costs = []

for j in range(1, 5):
    network = FFNN (network_structure)
    
    costs = network.train_network(training_data=x, training_labels=y, epochs=number_of_epochs,
                               batch_size=current_batch_size, learning_rate=current_learning_rate,
                               testing_data=x_test, testing_labels=y_test,
                               testing_frequency=testing_frequency, is_storing_costs=True)
    # Plotting the progress
    plt.plot(epochs, costs, label=format(current_learning_rate, '.5f'))
    current_learning_rate += learning_rate_increment
    # Clearing out the learned weights and biases
    del network

plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.legend(title="Learning Rates")
plt.show()

###################################################################################################
# 3D plot that tests both learning rates and batch sizes
###################################################################################################
#
## Constants
#number_of_epochs = 2000
#testing_frequency = 100
#network_structure = [1, 5, 2]
## Variables
#current_batch_size = 64
#current_learning_rate = 0
## For plotting
#learningRates = []
#batch_sizes = []
#costs = []
#
#while (current_batch_size <= 512):
#    for i in np.arange(0.00001, 0.0001, 0.00001):
#        network = FFNN(network_structure)
#        current_learning_rate = i
#        cost = network.train_network(training_data=x, training_labels=y, epochs=number_of_epochs,
#                                     batch_size = current_batch_size,
#                                     learning_rate=current_learning_rate, testing_data=x_test,
#                                     testing_labels=y_test, testing_frequency=testing_frequency,
#                                     is_storing_costs=False)
#        costs.append(cost)
#        learningRates.append(current_learning_rate)
#        batch_sizes.append(current_batch_size)
#        # Clearing out the learned weights and biases
#        del network
#    current_batch_size += current_batch_size
#
#costs = np.asarray(costs)
#learningRates = np.asarray(learningRates)
#batch_sizes = np.asarray(batch_sizes)
#
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(learningRates, batch_sizes, costs)
#
#ax.set_xlabel('Learning Rate')
#ax.set_ylabel('Batch Size')
#ax.set_zlabel('Costs after 2000 epochs')
#
#plt.show()
