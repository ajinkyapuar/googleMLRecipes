# from sklearn import datasets
# from sklearn.cross_validation import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
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
# clf = KNeighborsClassifier()
#
# clf.fit(X_train, y_train)
#
# predictions = clf.predict(X_test)
#
# print predictions
#
# print accuracy_score(y_test, predictions)

############################################################################################################################################

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        # print "asdf"
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # print "fdsa"
        predictions = []
        for row in X_test:
            # print row
            # label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# print "********************** BEGIN TRAIN DATA **********************"
#
# print X_train
#
# print "********************** END TRAIN DATA **********************"
#
# print "********************** BEGIN TRAIN TARGET **********************"
#
# print y_train
#
# print "********************** END TRAIN TARGET **********************"

clf = ScrappyKNN()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print predictions

print accuracy_score(y_test, predictions)