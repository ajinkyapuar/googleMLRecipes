from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)


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

clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print predictions

print accuracy_score(y_test, predictions)

############################################################################################################################################
