from Gate import *
from Unit import UnitCreator

class Layer(object):
    """docstring for Layer."""
    def __init__(self, ucreator, size):
        super(Layer, self).__init__()
        self.ucreator = ucreator
        self.size = size
        self.combiners = []
        self.activations = []

        for i in range(self.size):
            # Create the combiner and activation gate pair
            self.combiners.append(CombineGate(ucreator))
            self.activations.append(SigmoidGate(ucreator))
            self.activations[i].bind_gate_input(self.combiners[i])

    def bind_gate_input(self, gate):
        self.bind_unit_input(gate.output_unit)

    def bind_unit_input(self, unit):
        for i in range(self.size):
            self.combiners[i].bind_unit_input(unit)

    def forward(self):
        for i in range(self.size):
            self.combiners[i].forward()
            self.activations[i].forward()

    def backward(self):
        for i in range(self.size):
            self.activations[i].backward()
            self.combiners[i].backward()


if __name__ == '__main__':
    ucreator = UnitCreator()
    x = ucreator.new_unit(10., 0., 'x')
    layer = Layer(ucreator, 10)
    layer.bind_unit_input(x)
    cgate = CombineGate(ucreator)
    for actgate in layer.activations:
        cgate.bind_gate_input(actgate)
    layer.forward()
    cgate.forward()
    ucreator.zero_grad()
    cgate.set_grad(1)
    cgate.backward()
    layer.backward()

    ucreator.print_nodes()
