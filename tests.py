import unittest
from Gate import *
from Unit import *
from NN import Layer, LayerNetwork
from math import exp

class TestLayerNetwork(unittest.TestCase):
    def test_layer_network(self):
        n_inputs = 3
        layers = [1, 1]
        layernetwork = LayerNetwork(n_inputs, layers)

        self.assertEqual(len(layernetwork.ucreator.units),
            n_inputs # input units
            + n_inputs * layers[0] # parameters combining inputs
            + sum(layers[i-1]*layers[i] for i in range(1, len(layers))) # combining between gates
            + sum(layers) * 2 # The output units of each layer
            + layers[-1] # The combining parameters into the last combiner
            + 1
            )

        output = layernetwork.predict([1 for i in range(n_inputs)])
        self.assertEqual(output, 0)

        for unit in layernetwork.ucreator:
            unit.value = 1.

        def sigmoid(x):
            return 1. / (1. + exp(-x))

        output = layernetwork.predict([1 for i in range(n_inputs)])
        self.assertEqual(output, sigmoid(sigmoid(n_inputs)))

class TestLayers(unittest.TestCase):
    def test_layer(self):
        ucreator = UnitCreator()
        layer_size = 3
        n_inputs = 2

        layer = Layer(ucreator, layer_size)
        self.assertEqual(len(ucreator.units), layer_size * 2)

        # Bind some input values
        for i in range(n_inputs):
            layer.bind_unit_input(ucreator.new_unit(i, 0., "i%d" % i))

        # With all the inputs to zero, should be 0.5 (sigmoid function)
        layer.forward()
        for gate in layer.activations:
            self.assertEqual(gate.get_value(), 0.5)



class TestGates(unittest.TestCase):

    def test_combine(self):
        ucreator = UnitCreator()
        cgate = CombineGate(ucreator)

        # Create and bind the inputs
        n_inputs = 5
        for i in range(n_inputs):
            cgate.bind_unit_input(ucreator.new_unit(i, 0., 'i%d' % i))

        # Check with hade made the correct amounts
        self.assertEqual(len(ucreator.units), n_inputs * 2 + 1)

        # Run forwards, should be zero
        cgate.forward()
        self.assertEqual(cgate.output_unit.value, 0)

        # Now set the cgate values to one
        for unit in cgate.input_params:
            unit.value = 1

        # Should now be the sum of the input values (parameters are zero)
        cgate.forward()
        self.assertEqual(sum(range(n_inputs)), cgate.output_unit.value)

        # Check the backpropagation. The gradient on the the parameters will be
        # just the product of the gradient and the current input value
        pull = 3.5
        cgate.output_unit.grad = pull
        cgate.backward()
        for i in range(n_inputs):
            self.assertEqual(cgate.input_params[i].grad, i * pull)

if __name__ == '__main__':
    unittest.main()
