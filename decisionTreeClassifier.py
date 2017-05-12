############################################################################################################################################
# smooth : 0
# Bumpy : 1

############################################################################################################################################

# apple : 0
# orange : 1

############################################################################################################################################

# from sklearn import tree
#
# features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#
# labels = [0, 0, 1, 1]
#
# clf = tree.DecisionTreeClassifier()
#
# clf.fit(features, labels)
#
# print clf.predict([[180, 1]])
# # print clf.predict([[180, 1], [110, 1]])

############################################################################################################################################

# from sklearn import tree
# from sklearn.datasets import load_iris
#
# iris = load_iris()
#
# print iris.feature_names
# print iris.target_names
#
# # print iris.data
# # print iris.target
#
# clf = tree.DecisionTreeClassifier()
#
# clf.fit(iris.data, iris.target)
#
# print clf.predict([[5.5, 3.5, 4.1, 1.6]])


############################################################################################################################################

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

from sklearn.externals.six import StringIO
import pydot

iris = load_iris()

test_idx = [0, 50, 100]

# print "********************** BEGIN IRIS DATA **********************"
# print iris.data
# print "********************** END IRIS DATA **********************"
#
# print "********************** BEGIN IRIS TARGET **********************"
# print iris.target
# print "********************** END IRIS TARGET **********************"

# Training Data
train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx, axis=0)

# print "********************** BEGIN TRAIN DATA **********************"
#
# print train_data
#
# print "********************** END TRAIN DATA **********************"
#
# print "********************** BEGIN TRAIN TARGET **********************"
#
# print train_target
#
# print "********************** END TRAIN TARGET **********************"

# Testing Data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# print test_data
# print test_target
#
clf = tree.DecisionTreeClassifier()

clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

dot_data = StringIO()
# print dot_data
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names,
                     filled=True, rounded=True, impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# open -a preview iris.pdf

############################################################################################################################################


# print(__doc__)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.datasets import load_iris
# from sklearn.tree import DecisionTreeClassifier
#
# # Parameters
# n_classes = 3
# plot_colors = "bry"
# plot_step = 0.02
#
# # Load data
# iris = load_iris()
#
# for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
#                                 [1, 2], [1, 3], [2, 3]]):
#     # We only take the two corresponding features
#     X = iris.data[:, pair]
#     y = iris.target
#
#     # Train
#     clf = DecisionTreeClassifier().fit(X, y)
#
#     # Plot the decision boundary
#     plt.subplot(2, 3, pairidx + 1)
#
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                          np.arange(y_min, y_max, plot_step))
#
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
#     plt.xlabel(iris.feature_names[pair[0]])
#     plt.ylabel(iris.feature_names[pair[1]])
#     plt.axis("tight")
#
#     # Plot the training points
#     for i, color in zip(range(n_classes), plot_colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
#                     cmap=plt.cm.Paired)
#
#     plt.axis("tight")
#
# plt.suptitle("Decision surface of a decision tree using paired features")
# plt.legend()
# plt.show()

############################################################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
#
# greyhounds = 500
# labs = 500
#
# grey_height = 28 + 4 * np.random.randn(greyhounds)
# lab_height = 24 + 4 * np.random.randn(greyhounds)
#
# # # print "********************** BEGIN GREY HEIGHT **********************"
#
# print grey_height
#
# # # print "********************** END GREY HEIGHT **********************"
#
# # # print "********************** BEGIN LAB HEIGHT  **********************"
#
# print lab_height
#
# # # print "********************** END LAB HEIGHT **********************"
#
# plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
# plt.show()

############################################################################################################################################

# Feature selection: think as if u are the classifier (thought experiment). How many u would need to solve the problem
# avoid useless, redundant,  features
# independent features are the best
# features should be easy to understand
# ideal featues are informative, independent, simple

############################################################################################################################################

# from sklearn import datasets
# from sklearn.cross_validation import train_test_split
# from sklearn import tree
# from sklearn.metrics import accuracy_score
#
# iris = datasets.load_iris()
#
# X = iris.data
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
#
#
# # print "********************** BEGIN TRAIN DATA **********************"
# #
# # print X_train
# #
# # print "********************** END TRAIN DATA **********************"
# #
# # print "********************** BEGIN TRAIN TARGET **********************"
# #
# # print y_train
# #
# # print "********************** END TRAIN TARGET **********************"
#
# clf = tree.DecisionTreeClassifier()
#
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
#
# print predictions
#
# print accuracy_score(y_test, predictions)


############################################################################################################################################

############################################################################################################################################
