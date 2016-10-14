from Unit import Unit, UnitCreator
from Gate import MultiplyGate, AddGate

class SVM(object):
    """
    This is a simple example of an SVM model being implemented using this neural
    network framework.

    The function we are attempting to fit is f(x, y) = ax + by + c

    To create this function we have:
      * 3 gates (ax, bx, ax + bx + c)
    """
    def __init__(self):
        super(SVM, self).__init__()
        # Keep track of all the units in this model
        self.ucreator = UnitCreator()

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

    def forward(self):
        self.mgate_ax.forward()
        self.mgate_by.forward()
        self.agate.forward()

    def __backward__(self):
        self.agate.backward()
        self.mgate_by.backward()
        self.mgate_ax.backward()

    def predict(self, x, y):
        self.mgate_ax.set_input_value('x', x)
        self.mgate_by.set_input_value('y', y)
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

        # Regularisation, bring the parameters back towards zero
        for unit in self.ucreator:
            unit.grad -= unit.value

        self.apply_gradient()

    def apply_gradient(self, step_size = 0.1):
        for unit in self.ucreator:
            unit.value += unit.grad * step_size

if __name__ == '__main__':
    svm = SVM()
    svm.ucreator.print_nodes()
    svm.train(1, 2, 1)
    svm.ucreator.print_nodes()
