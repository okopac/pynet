import logging

from Unit import Unit
from Circuit import Circuit

class SVM(object):
    """docstring for SVM."""
    def __init__(self, step_size = 0.1):
        super(SVM, self).__init__()

        self.abc = [Unit(i, 0.) for i in range(3)]
        self.circuit = Circuit()

        self.step_size = step_size

    def forward(self, x, y):
        self.output = self.circuit.forward(*self.abc, Unit(x, 0), Unit(y, 0))
        return self.output

    def backward(self, label):
        pull = 0
        # If the label is +1, the we should have a postitive return value.
        # If not, drive upwards
        if label == 1 and self.output.value < 1:
            pull = 1
        if label == -1 and self.output.value > -1:
            pull = -1

        for u in self.abc:
            u.grad = 0

        self.circuit.backward(pull)

        # Regularisation, bring the parameters back towards zero
        self.abc[0].grad += -self.abc[0].value
        self.abc[1].grad += -self.abc[1].value

    def predict(self, x, y):
        result = self.forward(x, y)
        return 1 if result.value > 0 else -1

    def train(self, x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.update_parameters()

    def update_parameters(self):
        for i in range(len(self.abc)):
            self.abc[i].value += self.abc[i].grad * self.step_size

if __name__ == '__main__':
    svm = SVM()
    data = [
        [1.2, 0.7], [-0.3, -0.5], [3.0, 0.1], [-0.1, -1.0], [-1.0, 1.1], [2.1, -3]
    ]
    labels = [1, -1, 1, -1, -1, 1]
    def eval_model(data, labels, model):
        return sum([model.predict(*d) == l for d, l in zip(data, labels)]) * 100.0 / len(labels)

    for i in range(100):
        for d, l in zip(data, labels):
            svm.train(*d, l)
            print(svm.abc)

    print(eval_model(data, labels, svm))
