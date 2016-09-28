from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC

X,y = datasets.make_classification(n_samples=1000, n_features=10)

X1,y1 = datasets.make_classification(n_samples=1000, n_features=10)

clsAll=OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
clsOne=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X1, y1)

print("One vs all cost= %f" % clsAll.score(X,y))

print("One vs one cost= %f" % clsOne.score(X1,y1))