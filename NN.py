from Gate import *
from Unit import UnitCreator
from SVM import Network
import logging

class Layer(object):
    """docstring for Layer."""
    def __init__(self, ucreator, size, ActivationGate = SigmoidGate):
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
    def __init__(self, n_inputs, layers=[3], ActivationGate=SigmoidGate):
        super(LayerNetwork, self).__init__()

        # Create the hidden layers
        self.layers = []
        for lsize in layers:
            self.layers.append(Layer(self.ucreator, lsize, ActivationGate=ActivationGate))

        # Bind the layers together
        for i in range(1, len(self.layers)):
            self.layers[i].bind_layer_input(self.layers[i-1])

        # Create the inputs, and bind them to the first layer
        self.inputs = [self.ucreator.new_unit(0., 0., 'i%d' % i) for i in range(n_inputs)]
        for u in self.inputs:
            self.layers[0].bind_unit_input(u)

        # Bind outputs to the last layer. Use a combiner to bind them together
        self.last_combiner = CombineGate(self.ucreator)
        for actgate in self.layers[-1].activations:
            self.last_combiner.bind_gate_input(actgate)

    def train(self, input_values, label):
        output = self.predict(input_values)

        self.ucreator.zero_grad()

        pull = 0
        if label == 1 and output < 1:
            pull = 1.
        elif output > -1:
            pull = -1.

        logging.debug("Setting pull to %f" % pull)
        self.last_combiner.set_grad(pull)
        self.backward()

        # Regularisation, bring the parameters back towards zero
        for unit in self.ucreator:
            unit.grad -= unit.value

        self.apply_gradient(step_size=0.01)

    def forward(self):
        for layer in self.layers:
            layer.forward()
        self.last_combiner.forward()

    def backward(self):
        self.last_combiner.backward()
        for layer in reversed(self.layers):
            layer.backward()

    def predict(self, input_values):
        assert(len(input_values) == len(self.inputs))
        for iv, iu in zip(input_values, self.inputs):
            iu.value = iv
        self.forward()
        return self.last_combiner.get_value()


if __name__ == '__main__':
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    lnetwork = LayerNetwork(2, [3, 3, 3])
    data = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    labels = np.array([1, -1, -1, 1])
    for i in range(100):
        for d, l in zip(data, labels):
            print(lnetwork.predict([1, 1]))
            lnetwork.train(d, l)
