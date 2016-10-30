import numpy as np

def sigmoid(x, der=False):
    return sigmoid(x) * (1.0 - sigmoid(x)) if der else 1 / (1 + np.exp(-x))

def simpleerror(actual, predicted):
    return actual - predicted

def tanh(x, der=False):
    return 1.0 - x**2 if der else np.tanh(x)
