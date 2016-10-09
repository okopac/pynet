import logging

from Gate import MultiplyGate
from Unit import Unit

logging.basicConfig()

inputs = [Unit(1., 0.), Unit(2., 0.)]

xy = Unit(0., 0.)

mgate1 = MultiplyGate()

def run_forward():
    global xy, inputs
    xy = mgate1.forward(inputs[0], inputs[1])
    logging.warn("xy value is %d" % xy.value)


def run_backward():
    global xy, inputs
    xy.grad = 1
    mgate1.backward()

STEP_SIZE = 0.1

for i in range(10):
    run_forward()
    run_backward()
    grad_vector = [unit.grad for unit in inputs]
    logging.warn("Gradient vector is %s" % grad_vector)
    for unit in inputs:
        unit.value += unit.grad * STEP_SIZE
