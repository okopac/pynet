import logging

from Unit import Unit

class MultiplyGate(object):
    """docstring for MultiplyGate."""
    def __init__(self):
        super(MultiplyGate, self).__init__()

    def forward(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
        self.uout = Unit(u1.value * u2.value, 0.)
        return self.uout

    def backward(self):
        # Use the chain rule to pass back the gradient to the inputs
        logging.debug("MultiplyGate passing back gradient %f" % self.uout.grad)
        self.u1.grad = self.uout.grad * self.u1.value
        self.u2.grad = self.uout.grad * self.u2.value

class AddGate(object):
    """docstring for AddGate."""
    def __init__(self):
        super(AddGate, self).__init__()

    def forward(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
        self.uout = Unit(u1.value + u2.value, 0.)
        return self.uout

    def backward(self):
        # The gradient is just passed directly back in the add gate
        logging.debug("Add gate passing back gradient %f" % self.uout.grad)
        self.u1.grad = self.uout.grad
        self.u2.grad = self.uout.grad
