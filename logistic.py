# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
from os import X_OK
dataset=pd.read_csv('breast_cancer.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# Data Overview
from os import X_OK
dataset=pd.read_csv('breast_cancer.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# Checking for null values
print("\nMissing values in the dataset:")
print(dataset.isnull().sum())

# Exploratory data analysis
#Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(dataset)
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred=classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

# Computing the accuracy with k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard deviation: {:.2f} %".format(accuracies.std()*100))

