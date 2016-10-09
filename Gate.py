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
        self.u1.grad = self.uout.grad * self.u1.value
        self.u2.grad = self.uout.grad * self.u2.value
