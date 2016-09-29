#
# Adapted from David Julian - Designing Machine Learning Systems in Python under the Fair Use Doctrine of US Copyright law
# (used for research)
#
import sys
import numpy
from scipy.special import expit


class SampleNeuralNetwork:


    def __init__(self, outputNodes, numFeatures, hiddenNodes=30, SL1=0.0, SL2=0.2, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, shuff=True, batch=1, state=None):
        numpy.random.seed(state)
        self.outputNodes = outputNodes
        self.numFeatures = numFeatures
        self.hiddenNodes = hiddenNodes
        self.weightMatrix1, self.weightMatrix2 = self._create_weights()
        self.sl1 = SL1
        self.sl2 = SL2
        self.time = epochs
        self.eta = eta
        self.al = alpha
        self.decrement = decrease_const
        self.shuffle = shuff
        self.batches = batch

    def _label(self, y, k):
        onehot = numpy.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _create_weights(self):
        """Initialize weights with small random numbers."""
        w1 = numpy.random.uniform(-1.0, 1.0, size=self.hiddenNodes * (self.numFeatures + 1))
        w1 = w1.reshape(self.hiddenNodes, self.numFeatures + 1)
        w2 = numpy.random.uniform(-1.0, 1.0, size=self.outputNodes * (self.hiddenNodes + 1))
        w2 = w2.reshape(self.outputNodes, self.hiddenNodes + 1)
        return w1, w2

    def sigmoid_fun(self, z):
        return expit(z)

    def sigmoid_grad(self, z):
        sg = self.sigmoid_fun(z)
        return sg * (1 - sg)

    def addBias(self, X, how='column'):
        if how == 'column':
            X_new = numpy.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = numpy.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def feedforwards(self, X, w1, w2):
        a1 = self.addBias(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self.sigmoid_fun(z2)
        a2 = self.addBias(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self.sigmoid_fun(z3)
        return a1, z2, a2, z3, a3

    def PL2(self, lambda_, w1, w2):
        return (lambda_ / 2.0) * (numpy.sum(w1[:, 1:] ** 2) + numpy.sum(w2[:, 1:] ** 2))

    def PL1(self, lambda_, w1, w2):
        return (lambda_ / 2.0) * (numpy.abs(w1[:, 1:]).sum() + numpy.abs(w2[:,1:]).sum())

    def cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (numpy.log(output))
        term2 = (1 - y_enc) * numpy.log(1 - output)
        cost = numpy.sum(term1 - term2)
        L1_term = self.PL1(self.sl1, w1, w2)
        L2_term = self.PL2(self.sl2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        sigma3 = a3 - y_enc
        z2 = self.addBias(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self.sigmoid_grad(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (self.sl1 + self.sl2))
        grad2[:, 1:] += (w2[:, 1:] * (self.sl1 + self.sl2))
        return grad1, grad2

    def predict(self, X):
        if len(X.shape) != 2:
            raise AttributeError('')
        a1, z2, a2, z3, a3 = self.feedforwards(X, self.weightMatrix1, self.weightMatrix2)
        y_pred = numpy.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._label(y, self.outputNodes)
        delta_w1_prev = numpy.zeros(self.weightMatrix1.shape)
        delta_w2_prev = numpy.zeros(self.weightMatrix2.shape)
        for i in range(self.time):
            self.eta /= (1 + self.decrement * i)
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.time))
            sys.stderr.flush()
            if self.shuffle:
                idx = numpy.random.permutation(y_data.shape[0])
            X_data, y_data = X_data[idx], y_data[idx]
            mini = numpy.array_split(range(y_data.shape[0]), self.batches)
            for idx in mini:
                a1, z2, a2, z3, a3 = self.feedforwards(X[idx], self.weightMatrix1, self.weightMatrix2)
                cost = self.cost(y_enc=y_enc[:, idx],
                                 output=a3,
                                 w1=self.weightMatrix1,
                                 w2=self.weightMatrix2)
                self.cost_.append(cost)
            # compute gradient via backpropagation
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2,
                                                  a3=a3, z2=z2,
                                                  y_enc=y_enc[:, idx],
                                                  w1=self.weightMatrix1,
                                                  w2=self.weightMatrix2)
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.weightMatrix1 -= (delta_w1 + (self.al * delta_w1_prev))
                self.weightMatrix2 -= (delta_w2 + (self.al * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        return self