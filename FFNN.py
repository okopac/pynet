import functions
import numpy as np

class FFNN():
    def __init__(self, insize = 2, outsize = 1, layers=[2, 2], activation=functions.sigmoid, error_func=functions.simpleerror, learning_rate = 0.3, random_seed=None):
        if random_seed != None:
            np.random.seed(random_seed)
        self.activation = activation
        self.layers = [insize] + layers + [outsize]
        self.params = []
        self.learning_rate = learning_rate
        self.error_func = error_func

        self.values = []
        #
        #  x     x y C     1a
        #  y  .  x y C  =  1b
        #        x y C     1c
        #
        #  1a    1a 1b 1c CC
        #  1b .
        #  1c
        #
        for i in range(1, len(self.layers)):
            # Each transformation takes the vector of size previous layer to the next layer size.
            # However we also need to add a constant at each layer
            # Rather than doing this seperately, if we extend the previous layer by 1 element (with the value 1)
            # Then we can perform a matrix multiplication to map to the next layer size
            self.params.append(np.random.random((self.layers[i - 1] + 1, self.layers[i])) - 0.5)

    def forward(self, x):
        # We build an array of the values at each layer
        # This will be the same size as the layers
        # Start with the input
        self.values = [np.append(np.array(x), 1)]
        for i, p in enumerate(self.params):
            # Create the next layer by dot product with the parameter matrix
            # Note we add the constant 1, as the parameter matrix contains a constant
            output = np.dot(self.values[i], p)
            # Run the activation function to map these values
            act = self.activation(output)
            # Then add to the runnning values total
            if i == len(self.params) - 1:
                self.values.append(act)
            else:
                self.values.append(np.append(act, 1))

        # We return the output, which is the last layer
        return self.values[-1]

    def train(self, x, y):
        # The input vector must be the same size as the first matrix (minus the constant)
        assert(len(x) == self.params[0].shape[0] - 1)
        # The output sizes should match
        assert(len(y) == self.params[-1].shape[1])

        # Get the output
        res = self.forward(x)

        # Calculate the error
        error = self.error_func(y, self.values[-1])

        # Start building the errors
        # We start by taking the error, and back propagating through the activation function
        self.errors = [error * self.activation(self.values[-1], True)]

        # Then we
        last = True
        for l in range(len(self.values) - 2, 0, -1):
            # Take the previous error, and backprop through the parameter matrix
            # We take the transverse matrix here to invert the operation
            # We then drop the last value, as this is the 1 that we appending in the forward model
            if l == len(self.values) - 2:
                back_prop_errors = self.errors[-1].dot(self.params[l].T)
            else:
                back_prop_errors = self.errors[-1][:-1].dot(self.params[l].T)

            # Apply the activation function deriviative to pass back to the next layer
            # Remember the implicit 1 value at each layer
            self.errors.append(back_prop_errors * self.activation(self.values[l], True))

        self.errors.reverse()

        for i in range(len(self.params)):
            layer = np.atleast_2d(self.values[i])
            error = np.atleast_2d(self.errors[i][:-1])
            if i == len(self.params) - 1:
                error = np.atleast_2d(self.errors[i])
            self.params[i] += self.learning_rate * layer.T.dot(error)
