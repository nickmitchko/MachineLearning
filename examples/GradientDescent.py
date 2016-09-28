import numpy as numpy
import matplotlib.pyplot as pyplot
import random

def gd(x, y, a, iterations):
    xPrime = x.transpose()  # The transposition of matrix x (i.e. the elements' coordinates are swapped; x(1,3) becomes xPrime(3,1))
    k, l = numpy.shape(x)   # returns the size of the matrix x
    t = numpy.ones(l)       # create a vector of size n
    cost = 0
    for j in range(0, iterations):
        dotProduct = numpy.dot(x, t)    # The dot product of x and a same size vector filled with ones
        diff = dotProduct - y           # our "loss"
        cost = numpy.sum(diff ** 2)/2*k     # Cost function of our loss
        grad = numpy.dot(xPrime, diff) / k  # Measuring the gradient of our movement along the curve
        t -= a * grad   # update our theta value
    return t, cost

def generateDummyData(numberOfPoints, bias, variance):
    axis0 = numpy.zeros(shape=(numberOfPoints, 2))
    axis1 = numpy.zeros(shape=numberOfPoints)
    for j in range(0, numberOfPoints):
        axis0[j][0] = 1
        axis0[j][1] = j
        axis1[j] = (j + bias) + random.uniform(0, 1) * variance
    return axis0, axis1

x, y = generateDummyData(50, 25, 12)
iteration = 10000
alpha = 0.002
th, minCost = gd(x, y, alpha, iteration)

pyplot.scatter(x[...,1], y)
pyplot.plot(x[...,1], [th[0] + th[1]*xi for xi in x[...,1]])
pyplot.show()