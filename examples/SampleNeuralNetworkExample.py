from sklearn import datasets
import matplotlib.pyplot as plt
from examples import SampleNeuralNetwork

iris = datasets.load_iris()
X=iris.data
y=iris.target
nn= SampleNeuralNetwork.SampleNeuralNetwork(3, X.shape[1],hiddenNodes=50, epochs=100, alpha=.001)
nn.fit(X,y)
plt.plot(range(len(nn.cost_)),nn.cost_)
plt.show()