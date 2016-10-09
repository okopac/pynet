class Unit(object):
    def __init__(self, value=None, grad=None):
        self.value = value
        self.grad = grad

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Value: %f, Grad: %f" % (self.value, self.grad)
