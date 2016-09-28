import sys
import numpy
from scipy.special import expit

class SampleNeuralNetwork(object):

    def __init__(self, outputNodes, numFeatures, hiddenNodes=30, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):
        numpy.random.seed(random_state)
        self.n_output = outputNodes
        self.n_features = numFeatures
        self.n_hidden = hiddenNodes
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        onehot = numpy.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot