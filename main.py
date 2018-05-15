from src.model.classifier import *
import chainer
import numpy as np

def format(batch):
    xs = np.array([*map(lambda x: x[0], batch)])
    ts = np.array([*map(lambda x: x[1], batch)])
    return xs, np.eye(10)[ts]

def main():
    train, test = chainer.datasets.get_mnist()
    model = Model([784, 100, 10])
    batchsize = 1000
    for epoch in range(50):
        assert len(train) % batchsize == 0, "batchsize must be a divisor of trainsize: {}".format(len(train))
        for batch in chainer.iterators.SerialIterator(train, batchsize, repeat=False):
            xs, ts = format(batch)
            model(np.array(xs), np.array(ts))
        tx, tt = format(test)
        print("Epoch {0}: Accuracy: {1}".format(epoch, model.accuracy(tx, tt)))

if __name__ == '__main__':
    main()
