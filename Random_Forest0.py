import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})


path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names=headernames)
print(dataset.head())
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

##Split Train record 70% and 30%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y , test_size = 0.20)

##train the model
from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier (n_estimators = 50)
classifier.fit(X_train,Y_train)

##Predict
y_pred = classifier.predict(X_test)
print(y_pred)

print(classifier.estimators_[0])

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, y_pred)
print(result)
result1 = classification_report(Y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,y_pred)
print("Accuracy:",result2)
