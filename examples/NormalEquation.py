import numpy

X = numpy.matrix('1 2 5 4 8 2 6 3; 2 5 1 4 7 8 5 3 ')
y = numpy.matrix('8; 12')
correction = 0.8
i = numpy.matrix(numpy.identity(X.shape[1]))
XX1 = (X.T.dot(X)+correction * i).I # (X transpose * X)^-1
XX1.dot(X.T).dot(y)

print(XX1)