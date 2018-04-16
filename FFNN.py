# This free software is licensed under GPL version 3 without any implied or
# explicit warranties. This copyright notice must be included in all
# non-machine-executable copies or derivative works of this software.
import numpy as np
import scipy.special

# Feed Forward Neural Network
class FFNN:

    def __init__(self, dimensions):
        # 0th layer doesn't have any weights or biases
        self.weights = [0]
        self.biases = [0]
        self.gradient_b = [0]
        self.gradient_w = [0]
        self.num_of_layers = len(dimensions)

        randomization_scale_factor = 0.01

        for i in range(1, self.num_of_layers):
            self.weights.append(np.random.randn(dimensions[i], dimensions[i - 1])
                * randomization_scale_factor)
            self.biases.append(np.random.randn(dimensions[i], 1)
                * randomization_scale_factor)
            self.gradient_w.append(np.zeros(self.weights[i].shape))
            self.gradient_b.append(np.zeros(self.biases[i].shape))


    def train_network(self, training_data, training_labels, epochs, batch_size, learning_rate,
                      testing_data=None, testing_labels=None, testing_frequency=0,
                      is_storing_costs=False):
        assert (training_data.shape[0] == self.weights[1].shape[1])
        assert (training_data.shape[1] == training_labels.shape[1])

        data_given = False
        epochs_since_test = 0

        if testing_data is not None and testing_data.size != 0 and \
                testing_labels is not None and testing_labels.size != 0 and \
                testing_frequency != 0:
            data_given = True
            cost = 0
            if (is_storing_costs):
                costs = []

        for epoch in range(epochs):
            # Shuffling the data
            p = np.random.permutation(training_data.shape[1])
            for i in range(training_data.shape[0]):
                training_data[i] = training_data[i][p]
            for i in range(training_labels.shape[0]):
                training_labels[i] = training_labels[i][p]

            # Now dividing the data into minibatches and calculate
            start_index = 0
            end_index = batch_size

            while (start_index < training_data[0].shape[0]):
                # Forward propagate and retrieve intermediary values for this batch
                weighted_inputs, activations = self.forward_propagate(training_data[:,
                                                                      start_index:end_index],
                                                                      True, self.sigmoid)
                # Calculating gradient vector
                self.back_propagate(weighted_inputs, activations, training_labels[:, start_index:
                                                                                      end_index])

                scale = learning_rate
                for i in range(1, self.num_of_layers):
                    self.weights[i] = np.subtract(self.weights[i], scale * self.gradient_w[i])

                # Prepare for next batch
                start_index = end_index
                end_index += batch_size

            # Testing data performance
            if data_given:
                epochs_since_test += 1
                if epochs_since_test == testing_frequency:
                    epochs_since_test = 0
                    result = self.forward_propagate(testing_data, False, self.sigmoid)
                    cost = np.sum(self.calculate_cost(result, testing_labels))
                    if(is_storing_costs):
                        costs.append(cost)

        # End of training, delete the
        if data_given:
            if is_storing_costs:
                return costs
            return cost

    def forward_propagate(self, data, is_storing_values, activation_function):
        """ Calculates forward propagation with the given data.

        Parameters:
            -- data: A 2D list of inputs. Each column denotes an
            instance(of any shape). Number of columns determine the sample size.
            -- is_storing_values: boolean value for storing intermediary
            elements. Should be False if you just want the output from the
            neural network. True if you want to use this for backpropagation.
            -- activation_function: Function pointer denoting the activation
            function for every neuron in the layer.

        Return values:
            if is_storing_values == False only the activations for the last
            layer will be returned: "activations". Otherwise;
            -- weighted_inputs: z = W * A + B for each layer.
            -- activations: Outputs from each layer the neural network.
        """
        assert (self.weights[1].shape[1] == data.shape[0])

        if is_storing_values:
            weighted_inputs = [0]
            activations = [data]  # First layer is the outputs
            # Loop through the layers of the NN
            for j in range(1, self.num_of_layers):
                weighted_inputs.append(self.calculate_weighted_input(j, activations[j-1]))
                activations.append(activation_function(weighted_inputs[j]))
            return weighted_inputs, activations
        else:
            # Input layer is the first "activation layer"
            activations = data
            # Loop through the layers of the NN
            for j in range(1, self.num_of_layers):
                activations = activation_function(self.calculate_weighted_input(j, activations))
            return activations

    def back_propagate(self, weighted_inputs, activations, labels):
        """ Calculates errors and backpropagates them accordingly """
        # Backward pass
        delta = self.calculate_cost_derivative(activations[-1], labels)
        delta = np.multiply(delta, self.sigmoid_derivative(activations[-1]))

        for i in range(1, self.num_of_layers):
            self.gradient_b[-i] = delta
            self.gradient_w[-i] = np.dot(delta, activations[-i-1].T)

            # Calculating delta layer 1:
            delta = np.dot(self.weights[-i].T, delta)
            delta = np.multiply(delta, self.sigmoid_derivative(activations[-i-1]))

    def calculate_weighted_input(self, layer, a_prev):
        """ Calculates weighted inputs with a vectorized calculation for a
        single layer. (A_current_layer = Weights * A_previous_layer + Biases)

        Parameters:
            -- layer: An integer denoting the layer number of the neural net.
            -- a_prev: A numpy array denoting the previous layer's outputs.

        Return values:
            -- z: Weight * a_prev + bias
        """
        assert (self.weights[layer].shape[1] == a_prev.shape[0])
        return np.dot(self.weights[layer], a_prev) + self.biases[layer]

    def calculate_cost(self, a, y):
        """ Calculates the cost according to the chosen function for the output
        layer. a is the same as y_hat and y is the expected output.
        """
        return self.calculate_quadratic_cost(a, y)

    def calculate_cost_derivative(self, a, y):
        """ Calculates the partial derivative for the chosen cost function for
        the output layer. a is the same as y_hat and y is the expected output.
        """
        return self.calculate_quadratic_cost_derivative(a, y)

    def calculate_quadratic_cost(self, y, y_hat):
        """ Calculates the quadratic cost function of the form:

            C = (1 / 2) * (Y - Y_HAT)^2

            Y being the expected output and Y_HAT being the output given by the
            neural network.

            Parameters:
                -- y: Numpy array of expected results shaped as (n, m).
                -- y_hat: Numpy array of produced results shaped as (n, m)
        """
        assert (y.shape[0] == y_hat.shape[0])
        assert (y.shape[1] == y_hat.shape[1])

        return (np.subtract(y, y_hat) ** 2) / 2

    def calculate_quadratic_cost_derivative(self, a, y):
        """ Calculates the partial derivative for the quadratic cost function
        for the output layer. a is the same as y_hat

        d C_x / d_a = (a - y)

        Parameters:
            -- a: Output of the activation functions for the final layer.
            -- y: Expected outputs.
        Returns:
            Float derivative of the cost prime denoted above.
        """
        return np.subtract(a, y)

    def sigmoid(self, param):
        """ derivative of sigmoid is 1/(1+exp(-x)) which is what expit does.
        """
        return scipy.special.expit(param)

    def sigmoid_derivative(self, param):
        return param * (1 - param)
