from Unit import Unit, UnitCreator
from Gate import MultiplyGate, AddGate

class Network(object):
    """docstring for network."""
    def __init__(self):
        super(Network, self).__init__()
        self.ucreator = UnitCreator()

    def train(self, input_values, label):
        raise NotImplementedError

    def apply_gradient(self, step_size = 0.1):
        for unit in self.ucreator:
            unit.value += unit.grad * step_size

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def predict(self, input_values):
        raise NotImplementedError

class SVM(Network):
    """
    This is a simple example of an SVM model being implemented using this neural
    network framework.

    The function we are attempting to fit is f(x, y) = ax + by + c

    To create this function we have:
      * 3 gates (ax, bx, ax + bx + c)
    """
    def __init__(self):
        super(SVM, self).__init__()

        # ax
        self.mgate_ax = MultiplyGate(self.ucreator)
        self.mgate_ax.bind_unit_input(self.ucreator.new_unit(0., 0., 'a'))
        self.mgate_ax.bind_unit_input(self.ucreator.new_unit(0., 0., 'x'))

        # bx
        self.mgate_by = MultiplyGate(self.ucreator)
        self.mgate_by.bind_unit_input(self.ucreator.new_unit(0., 0., 'b'))
        self.mgate_by.bind_unit_input(self.ucreator.new_unit(0., 0., 'y'))

        # ax + bx + c
        self.agate = AddGate(self.ucreator)
        self.agate.bind_unit_input(self.ucreator.new_unit(0., 0., 'c'))
        self.agate.bind_gate_input(self.mgate_ax)
        self.agate.bind_gate_input(self.mgate_by)

    def train(self, input_values, label):
        self.predict(input_values)

        self.ucreator.zero_grad()

        pull = 0
        if label == 1 and self.agate.get_value() < 1:
            pull = 1.
        if label == -1 and self.agate.get_value() > -1:
            pull = -1.

        self.agate.set_grad(pull)
        self.__backward__()

        # Regularisation, bring the parameters back towards zero
        for unit in self.ucreator:
            unit.grad -= unit.value

        self.apply_gradient()

    def forward(self):
        self.mgate_ax.forward()
        self.mgate_by.forward()
        self.agate.forward()

    def __backward__(self):
        self.agate.backward()
        self.mgate_by.backward()
        self.mgate_ax.backward()

    def predict(self, input_values):
        assert(len(input_values) == 2)
        self.mgate_ax.set_input_value('x', input_values[0])
        self.mgate_by.set_input_value('y', input_values[1])
        self.forward()
        return self.agate.get_value()


if __name__ == '__main__':
    svm = SVM()
    svm.ucreator.print_nodes()
    svm.train([1, 2], 1)
    svm.ucreator.print_nodes()
