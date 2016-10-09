import logging

from Gate import MultiplyGate, AddGate
from Unit import Unit

class Circuit(object):
    """docstring for Circuit."""
    def __init__(self):
        super(Circuit, self).__init__()

        self.mgate1 = MultiplyGate()
        self.mgate2 = MultiplyGate()

        self.agate1 = AddGate()
        self.agate2 = AddGate()

    def forward(self, a, b, c, x, y):
        # Our network looks like this:
        # f(x, y) = ax + by + c
        ax = self.mgate1.forward(a, x)
        by = self.mgate2.forward(b, y)

        ax_by = self.agate1.forward(ax, by)
        self.ax_by_c = self.agate2.forward(ax_by, c)

        return self.ax_by_c

    def backward(self, grad):
        self.ax_by_c.grad = grad
        self.agate2.backward()
        self.agate1.backward()
        self.mgate2.backward()
        self.mgate1.backward()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    c = Circuit()
    abc = [Unit(i, 0.) for i in range(3)]
    xy = [Unit(i, 0.) for i in range(2)]
    out = c.forward(*abc, *xy)
    c.backward(1.)
    for unit in abc:
        print(unit)
