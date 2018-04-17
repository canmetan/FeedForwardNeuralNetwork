# FeedForwardNeuralNetwork

Implementation of a customizable feed-forward neural network from scratch.

![2D Plot](https://github.com/canmetan/FeedForwardNeuralNetwork/blob/master/images/2dplot.png?raw=true)

![3D Plot](https://github.com/canmetan/FeedForwardNeuralNetwork/blob/master/images/3dplot.png?raw=true)

### Requirements
* Python 3.5.3 or newer.
* Numpy (for matrix operations & ndarrays)
* Scipy (for an accurate sigmoid function)

### Usage:
Simply execute "test.py" to see it in action. "FFNN.py" contains the network itself.
* The 2D plot has fixed batch size and tests various learning rates
* The 3D plot tests out various batch sizes and learning rates at the same time.
Comment out their respective sections to see their output.

### Explanation
The example neural net has 1 neuron on the input layer (receives x) and tries to generalize for the two equations (y1 and y2) for its output layer:

<a href="https://www.codecogs.com/eqnedit.php?latex=y1&space;=&space;x*\frac{sin(\frac{2\pi}{2x&plus;1})}{10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y1&space;=&space;x*\frac{sin(\frac{2\pi}{2x&plus;1})}{10}" title="y1 = x*\frac{sin(\frac{2\pi}{2x+1})}{10}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=y2&space;=&space;x*\frac{cos(\frac{2\pi}{2x&plus;1})}{1000}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y2&space;=&space;x*\frac{cos(\frac{2\pi}{2x&plus;1})}{1000}" title="y2 = x*\frac{cos(\frac{2\pi}{2x+1})}{1000}" /></a>

<u>This implementation has following properties</u>;

* Stochastic gradient descent, with modifiable mini-batch size.
* Shuffles training data after each epoch.
* Modifiable network structure.
* Can perform tests on a specified interval with the testing set and returns the cost (when a testing set is given).
* Weights and biases are randomly initialized.
* Customizable to contain other activation functions or cost functions (currently supporting sigmoid as the non-linear activation function and quadratic cost).
* Flexible to contain any type of input and output structure.
* Entire repo is GPL v3.