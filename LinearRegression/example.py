import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from LinearRegression import LinearRegression

X = np.array([1,2,3,4,5,6,7,8,9,10])#.reshape(-1,1)
y = np.array([1,2,3,4,5,6,8,8,8,10])#.reshape(-1,1)

# X = preprocessing.scale(X) 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


clf = LinearRegression()
# clf = svm.SVR()
clf.fit(X_train,y_train)


# accuracy = clf.score(X_test,y_test)
# print(accuracy)

