from Unit import Unit, UnitCreator
from Gate import MultiplyGate, AddGate

class Network(object):
    """docstring for Network."""
    def __init__(self, unitcreator):
        super(Network, self).__init__()
        self.unitcreator = unitcreator

    def apply_gradient(self, step_size = 0.1):
        for unit in unitcreator:
            unit.value += unit.grad * step_size

class SVM(object):
    """docstring for TestNetwork.
    The function we are doing is f(x, y) = ax + by + c
    """
    def __init__(self):
        super(SVM, self).__init__()
        self.ucreator = UnitCreator()
        self.gates = []
        self.mgate1 = MultiplyGate(self.ucreator)
        self.mgate1.bind_unit_input(self.ucreator.new_unit(0., 0., 'a'))
        self.mgate1.bind_unit_input(self.ucreator.new_unit(0., 0., 'x'))
        self.mgate2 = MultiplyGate(self.ucreator)
        self.mgate2.bind_unit_input(self.ucreator.new_unit(0., 0., 'b'))
        self.mgate2.bind_unit_input(self.ucreator.new_unit(0., 0., 'y'))
        self.agate = AddGate(self.ucreator)
        self.agate.bind_unit_input(self.ucreator.new_unit(0., 0., 'c'))
        self.agate.bind_gate_input(self.mgate1)
        self.agate.bind_gate_input(self.mgate2)

    def forward(self):
        self.mgate1.forward()
        self.mgate2.forward()
        self.agate.forward()

    def __backward__(self):
        self.agate.backward()
        self.mgate2.backward()
        self.mgate1.backward()

    def predict(self, x, y):
        self.mgate1.set_input_value('x', x)
        self.mgate2.set_input_value('y', y)
        self.forward()
        return self.agate.get_value()

    def train(self, x, y, label):
        self.predict(x, y)

        for u in self.ucreator:
            u.grad = 0
        pull = 0
        if label == 1 and self.agate.get_value() < 1:
            pull = 1.
        if label == -1 and self.agate.get_value() > -1:
            pull = -1.

        self.agate.set_grad(pull)
        self.__backward__()

        self.apply_gradient()

        # Regularisation, bring the parameters back towards zero
        #self.a.grad += -self.a.value
        #self.b.grad += -self.b.value

    def apply_gradient(self, step_size = 0.1):
        for unit in self.ucreator:
            unit.value += unit.grad * step_size

if __name__ == '__main__':
    svm = SVM()
    svm.ucreator.print_nodes()
    svm.train(1, 2, 1)
    svm.ucreator.print_nodes()
