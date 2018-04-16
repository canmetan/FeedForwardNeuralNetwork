# FeedForwardNeuralNetwork

Implementation of a customizable feed-forward neural network from scratch.


### Usage:
* Simply execute "test.py" to see an example.
* The example neural net receives a sequential input x and tries to generalize for the following two equations:

<a href="https://www.codecogs.com/eqnedit.php?latex=y1&space;=&space;x*\frac{sin(\frac{2\pi}{2x&plus;1})}{10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y1&space;=&space;x*\frac{sin(\frac{2\pi}{2x&plus;1})}{10}" title="y1 = x*\frac{sin(\frac{2\pi}{2x+1})}{10}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=y2&space;=&space;x*\frac{cos(\frac{2\pi}{2x&plus;1})}{1000}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y2&space;=&space;x*\frac{cos(\frac{2\pi}{2x&plus;1})}{1000}" title="y2 = x*\frac{cos(\frac{2\pi}{2x+1})}{1000}" /></a>

* The 2D plot has fixed batch size and tests various learning rates
* The 3D plot tests out various batch sizes and learning rates at the same time.

This implementation has following properties;

* Stochastic gradient descent, with modifiable mini-batch size.
* Shuffles testing data after each epoch
* Modifiable network structure
* Can perform tests on a specified interval with the testing set and returns the cost (when a testing set is given)
* Weights and biases are randomly initialized
* Customizable to contain other activation functions or cost functions (currently supporting sigmoid as the non-linear activation function and quadratic cost).
* Flexible to contain any type of input and output structure
* Entire repo is GPL v3.