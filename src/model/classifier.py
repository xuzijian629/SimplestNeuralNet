import chainer
import numpy as np
from src.functions.activation import *
from src.functions.loss import *

class Model:
    def __init__(self, layer_sizes, activation=sigmoid, activationdif=sigmoiddif, loss=mse, lossdif=msedif):
        self.weights = []
        self.biases = []
        self.n = len(layer_sizes) - 1
        for i in range(self.n):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            bias = np.ones(layer_sizes[i + 1]) / 10
            self.weights.append(weight)
            self.biases.append(bias)
        self.activation = activation
        self.activationdif = activationdif
        self.loss = loss
        self.lossdif = lossdif

    def __call__(self, xs, ts):
        self.forprop(xs)
        self.backprop(xs, ts)

    def predict(self, xs):
        self.forprop(xs)
        return np.eye(10)[np.argmax(self.ys[-1], axis=1)]

    def accuracy(self, xs, ts):
        ys = self.predict(xs)
        return np.sum((ys == ts).all(axis=1)) / len(ts)

    def forprop(self, xs):
        self.as_ = []
        self.ys = [xs]
        for i in range(self.n):
            a = np.dot(self.ys[-1], self.weights[i]) + self.biases[i]
            y = self.activation(a)
            self.as_.append(a)
            self.ys.append(y)

    # https://deepage.net/features/numpy-neuralnetwork-5.html
    def backprop(self, xs, ts, eta=1.0):
        dLdy = self.lossdif(self.ys[-1], ts)
        for i in range(self.n):
            if i > 0:
                dLdy = np.dot(dLdy, self.weights[self.n - i].T)
            dyda = self.activationdif(self.as_[self.n - 1 - i])
            dLda = dLdy * dyda
            dadw = self.ys[self.n - 1 - i].T
            self.biases[self.n - 1 - i] -= eta * np.sum(dLda, axis=0)
            self.weights[self.n - 1 - i] -= eta * np.dot(dadw, dLda)
