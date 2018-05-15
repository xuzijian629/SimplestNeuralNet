import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoiddif(x):
    f = sigmoid(x)
    return f * (1 - f)
