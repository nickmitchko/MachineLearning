#
# Modified from:
#
#       Designing Machine Learning Systems with Python - David Julian
#
# Note:
#
#       This file is not meant to infringe on the copyright of David Julian and is used under the fair use doctrine of US Copyright Law
#       which stipulates that ideas taken from copyrighted material may be reimplemented as long as it isn't directly copied

import numpy as numpy
import matplotlib.pyplot as pyplot
import random

def gradientDescent(x, y, a, iterations):
    xPrime = x.transpose()  # The transposition of vector x