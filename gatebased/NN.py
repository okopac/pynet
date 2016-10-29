from Gate import *
from Unit import UnitCreator
import logging

class Network(object):
    """docstring for network."""
    def __init__(self, step_size = 0.01, regularisation = True):
        super(Network, self).__init__()
        self.ucreator = UnitCreator()
        self.step_size = step_size
        self.regularisation = regularisation

    def train(self, input_values, label):
        raise NotImplementedError

    def apply_gradient(self):
        for unit in self.ucreator:
            unit.value += unit.grad * self.step_size

    def set_backprop(self, pull):
        raise NotImplementedError

    def train(self, input_values, label):
        output = self.predict(input_values)

        self.ucreator.zero_grad()

        pull = 0
        if label == 1 and self.combiner.get_value() < 1:
            pull = 1.
        elif label == -1 and self.combiner.get_value() > -1:
            pull = -1.
        elif not label in [1, -1]:
            raise Exception("Label must be 1 or -1 (%f)" % label)
        elif not output in [1, -1]:
            raise Exception("output must be 1 or -1 (%s)" % output)

        logging.debug("Setting pull to %f" % pull)
        # Set the output gradient, and back propagate
        self.set_backprop(pull)
        self.backward()

        # Regularisation, bring the parameters back towards zero
        if self.regularisation:
            raise NotImplementedError
            for unit in self.ucreator:
                unit.grad -= unit.value

        self.apply_gradient()

        # Return the pull, so the user knows if this was right or wrong
        return pull

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def predict(self, input_values):
        raise NotImplementedError

class Layer(object):
    """docstring for Layer."""
    def __init__(self, ucreator, size, ActivationGate = TanHGate):
        super(Layer, self).__init__()
        self.ucreator = ucreator
        self.size = size
        self.combiners = []
        self.activations = []

        for i in range(self.size):
            # Create the combiner and activation gate pair
            self.combiners.append(CombineGate(ucreator))
            self.activations.append(ActivationGate(ucreator))
            self.activations[i].bind_gate_input(self.combiners[i])

    def bind_gate_input(self, gate):
        self.bind_unit_input(gate.output_unit)

    def bind_unit_input(self, unit):
        for i in range(self.size):
            self.combiners[i].bind_unit_input(unit)

    def bind_layer_input(self, layer):
        for prev_gate in layer.activations:
            self.bind_gate_input(prev_gate)

    def forward(self):
        for i in range(self.size):
            self.combiners[i].forward()
            self.activations[i].forward()

    def backward(self):
        for i in range(self.size):
            self.activations[i].backward()
            self.combiners[i].backward()

class LayerNetwork(Network):
    """docstring for LayerNetwork."""
    def __init__(self, n_inputs, layers=[3], ActivationGate=TanHGate):
        super(LayerNetwork, self).__init__()

        # Create the hidden layers
        self.layers = []
        for lsize in layers:
            self.layers.append(Layer(self.ucreator, lsize, ActivationGate=ActivationGate))

        # Bind the layers together
        for i in range(1, len(self.layers)):
            self.layers[i].bind_layer_input(self.layers[i-1])

        # Create the inputs, and bind them to the first layer
        self.inputs = [self.ucreator.new_unit(name='i%d' % i) for i in range(n_inputs)]
        for u in self.inputs:
            self.layers[0].bind_unit_input(u)

        # Bind outputs to the last layer. Use a combiner to bind them together
        self.combiner = CombineGate(self.ucreator)
        for actgate in self.layers[-1].activations:
            self.combiner.bind_gate_input(actgate)

        # Set random variables for all the nodes
        self.ucreator.initialise_values()

    def set_backprop(self, pull):
        self.combiner.set_grad(pull)

    def forward(self):
        for layer in self.layers:
            layer.forward()
        self.combiner.forward()

    def backward(self):
        self.combiner.backward()
        # print(self.combiner.bias.grad)
        # for a in self.layers[-1].activations:
        #     print(a.output_unit)

        for layer in reversed(self.layers):
            layer.backward()

        # for a in self.layers[0].activations:
        #     print(a.output_unit)
        # raise NotImplementedError

    def predict(self, input_values):
        assert(len(input_values) == len(self.inputs))
        for i, iv in enumerate(input_values):
            self.inputs[i].value = iv
        self.forward()
        return 1 if self.combiner.get_value() >= 0 else -1


if __name__ == '__main__':
    import numpy as np
    import itertools
    logging.basicConfig(level=logging.INFO)
    lnetwork = LayerNetwork(2, [2, 8, 3])
    # Check the inter layer connections
    for l in range(1, len(lnetwork.layers)):
        for i, j in itertools.product(range(len(lnetwork.layers[l-1].activations)), range(len(lnetwork.layers[l].combiners))):
            assert(lnetwork.layers[l - 1].activations[i].output_unit in lnetwork.layers[l].combiners[j].inputs)
