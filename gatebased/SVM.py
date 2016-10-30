from Unit import Unit, UnitCreator
from Gate import CombineGate
from NN import Network

class SVM(Network):
    """
    This is a simple example of an SVM model being implemented using this neural
    network framework.

    The function we are attempting to fit is f(x, y) = ax + by + c
    """
    def __init__(self):
        super(SVM, self).__init__()
        self.x = self.ucreator.new_unit(name='x')
        self.y = self.ucreator.new_unit(name='y')
        self.combiner = CombineGate(self.ucreator)
        self.combiner.bind_unit_input(self.x)
        self.combiner.bind_unit_input(self.y)
        self.ucreator.initialise_values()

    def forward(self):
        self.combiner.forward()

    def backward(self):
        self.combiner.backward()

    def set_backprop(self, grad):
        self.combiner.set_grad(grad)

    def predict(self, input_values):
        self.x.value = input_values[0]
        self.y.value = input_values[1]
        self.forward()
        return 1 if self.combiner.get_value() >= 0 else -1


if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt

    def target_function(x, y):
        val = 3.5 * x - 9.8 * y + 0.2
        return 1 if val > 0 else - 1

    train_data = [(random.random(), random.random()) for i in range(10)]
    test_data = [(random.random(), random.random()) for i in range(10)]
    train_label = [target_function(x, y) for x, y in train_data]
    test_label = [target_function(x, y) for x, y in test_data]

    test_eval = []
    train_eval = []

    svm = SVM()
    svm.regularisation = False
    svm.step_size = 1

    def eval_model(model, data, labels):
        return sum([model.predict(d) == l for d, l in zip(data, labels)]) * 100.0 / len(labels)

    a, b, c = [], [], []

    for i in range(10000):
        a.append(svm.combiner.input_params[0].value)
        b.append(svm.combiner.input_params[1].value)
        c.append(svm.combiner.bias.value)
        for x, y in train_data:
            x, y = random.random(), random.random()
            pull = svm.train([x, y], target_function(x, y))
        train_eval.append(eval_model(svm, train_data, train_label))
        test_eval.append(eval_model(svm, test_data, test_label))

    svm.ucreator.print_nodes()
    #plt.plot(a, label='a')
    #plt.plot(b, label='b')
    #plt.plot(c, label='c')
    #plt.legend()
    #plt.show()

    plt.plot(train_eval, label='train')
    plt.plot(test_eval, label='test')
    plt.legend()
    plt.show()
