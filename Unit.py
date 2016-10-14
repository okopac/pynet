class Unit(object):
    """
    The unit object is the only class in the NN that stores parameters of the
    model. It has three components, value (used in forward propagation), grad
    (used in backpropagation) and name (used to help keep track of where the
    parameters are used).
    """
    def __init__(self, value = None, grad = None, name=None):
        super(Unit, self).__init__()
        self.value = value
        self.grad = grad
        self.name = name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.name != None:
            return "Value: %s, Grad: %s (%s)" % (self.value, self.grad, self.name)
        else:
            return "Value: %s, Grad: %s" % (self.value, self.grad)

class UnitCreator(object):
    """
    A class that is responsible for creating units. We do this to keep track of
    all the units that are in our network.
    """
    def __init__(self):
        super(UnitCreator, self).__init__()
        self.units = []

    def new_unit(self, *args):
        self.units.append(Unit(*args))
        return self.units[-1]

    def print_nodes(self):
        for u in self.units:
            print(u)

    def __iter__(self):
        for u in self.units:
            yield u
