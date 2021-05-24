X =  list(range(10))
print (X)
y = [x*x for x in X]
print(y)
import sklearn.model_selection as model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.75,test_size=0.25)
print ("X_train: ", X_train)
print ("y_train: ", y_train)
print ("X_test: ", X_test)
print ("y_test: ", y_test)
