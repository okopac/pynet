import logging

from Unit import Unit
from Circuit import Circuit

class SVM(object):
    """docstring for SVM."""
    def __init__(self, step_size = 0.01):
        super(SVM, self).__init__()

        self.a = Unit(1., 0.)
        self.b = Unit(-2., 0.)
        self.c = Unit(-1., 0.)

        self.circuit = Circuit()

        self.step_size = step_size

    def forward(self, x, y):
        self.output = self.circuit.forward(self.a, self.b, self.c, Unit(x, 0), Unit(y, 0))
        return self.output

    def backward(self, label):
        pull = 0.
        # If the label is +1, the we should have a postitive return value.
        # If not, drive upwards
        if label == 1 and self.output.value < 1:
            pull = 1.
        if label == -1 and self.output.value > -1:
            pull = -1.

        self.a.grad = 0.
        self.b.grad = 0.
        self.c.grad = 0.

        self.circuit.backward(pull)

        # Regularisation, bring the parameters back towards zero
        self.a.grad += -self.a.value
        self.b.grad += -self.b.value

    def predict(self, x, y):
        result = self.forward(x, y)
        return 1 if result.value > 0 else -1

    def train(self, x, y, label):
        res = self.forward(x, y)
        self.backward(label)
        logging.debug("Result from training (%s, %s, %d): %s" % (x, y, label, res))
        self.update_parameters()

    def update_parameters(self):
        self.a.value += self.a.grad * self.step_size
        self.b.value += self.b.grad * self.step_size
        self.c.value += self.c.grad * self.step_size

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    svm = SVM()
    print("f(x, y) = %fx + %fy + %f" % (svm.a.value, svm.b.value, svm.c.value))

    data = [
        [1.2, 0.7], [-0.3, 0.5], [-3, -1], [0.1, 1.0], [3.0, 1.1], [2.1, -3]
    ]
    labels = [1, -1, 1, -1, -1, 1]
    def eval_model(data, labels, model):
        return sum([model.predict(*d) == l for d, l in zip(data, labels)]) * 100. / len(labels)

    for i in range(401):
        for d, l in zip(data, labels):

            logging.info("f(x, y) = %fx + %fy + %f" % (svm.a.value, svm.b.value, svm.c.value))
            svm.train(*d, l)
            if i % 100 == 0:
                print(eval_model(data, labels, svm), i)
                print("  f(x, y) = %fx + %fy + %f" % (svm.a.value, svm.b.value, svm.c.value))
