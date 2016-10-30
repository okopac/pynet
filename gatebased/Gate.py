from Unit import UnitCreator
from math import exp
import logging

class Gate(object):
    """
    The gate is the main computational object in the neural net. This is the
    base class, and any gate is required to implement forward and backward, the
    process of propagating signal through the gate.

    When gates are initialised they create a new output unit, this is where they
    save thier output data. To add inputs, or parameters, the gate uses the
    unitcreator object. This is so we can keep track of all units that have been
    created in the network.
    """
    def __init__(self, ucreator):
        super(Gate, self).__init__()
        self.ucreator = ucreator
        self.inputs = []
        self.output_unit = self.ucreator.new_unit(name='out')

    def bind_gate_input(self, gate):
        self.bind_unit_input(gate.output_unit)

    def bind_unit_input(self, unit):
        self.inputs.append(unit)

    def set_grad(self, grad):
        self.output_unit.grad = grad

    def get_value(self):
        return self.output_unit.value

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def set_input_value(self, iname, value):
        for u in self.inputs:
            if iname == u.name:
                u.value = value

class CombineGate(Gate):
    """docstring for CombineGate."""
    def __init__(self, *args):
        super(CombineGate, self).__init__(*args)
        self.input_params = []
        self.bias = self.ucreator.new_unit(name='cbias')

    def bind_unit_input(self, unit):
        # Everytime something wants to add an input to this gate, we add an
        # additional parameter that is a weight
        super(CombineGate, self).bind_unit_input(unit)
        self.input_params.append(self.ucreator.new_unit(name='c%d' % len(self.input_params)))

    def forward(self):
        # Combine gate combines all inputs with a weighted parameters
        # i.e. f(x, y, z) = ax + by + cz
        self.output_unit.value = self.bias.value + sum((a.value * x.value for a, x in zip(self.inputs, self.input_params)))

    def backward(self):
        # Use the chain rule to pass back the gradient to the inputs
        # Multiple the output gradient, by the derivative of the function
        self.bias.grad = self.output_unit.grad
        for i in range(len(self.inputs)):
            self.input_params[i].grad += self.output_unit.grad * self.inputs[i].value
            self.inputs[i].grad += self.output_unit.grad * self.input_params[i].value

class SigmoidGate(Gate):
    """docstring for Sigmoid."""
    def __init__(self, *args):
        super(SigmoidGate, self).__init__(*args)

    def forward(self):
        assert(len(self.inputs) == 1)
        self.output_unit.value = self._sigmoid(self.inputs[0].value)

    def backward(self):
        self.inputs[0].grad += self.output_unit.grad * self.output_unit.value * (1 - self.output_unit.value)

    def _sigmoid(self, x):
        return 1. / (1. + exp(-x))


class TanHGate(Gate):
    """docstring for TanHGate."""
    def __init__(self, *args):
        super(TanHGate, self).__init__(*args)

    def forward(self):
        assert(len(self.inputs) == 1)
        self.output_unit.value = self._tanh(self.inputs[0].value)

    def backward(self):
        self.inputs[0].grad += self.output_unit.grad * (1 - self.output_unit.value ** 2)

    def _tanh(self, x):
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

if __name__ == '__main__':
    ucreator = UnitCreator()
    mgate1 = MultiplyGate(ucreator)
    mgate1.bind_unit_input(ucreator.new_unit(5., 1., 'input1'))
    mgate1.bind_unit_input(ucreator.new_unit(1., 1., 'input2'))
    mgate2 = MultiplyGate(ucreator)
    mgate2.bind_gate_input(mgate1)
    mgate2.bind_unit_input(ucreator.new_unit(9., 1.))
    print(mgate1.forward())
    print(mgate2.forward())
    mgate2.set_grad(1)
    mgate2.backward()
    mgate1.backward()
    print("after backprop")
    ucreator.print_nodes()
