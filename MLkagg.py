import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

sns.set()

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# store target variable of training data in a safe place
survived_train = df_train.Survived

#concaenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis =1), df_test])

# Impute missing numerical variables by putting in NA
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# encode male and female to numbers
data = pd.get_dummies(data, columns=['Sex'], drop_first = True)

# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]

#split data back into training/test sets
data_train = data.iloc[:891]
data_test = data.iloc[891:]

#transform data from dataframe to array
X = data_train.values
test = data_test.values
y = survived_train.values

#making decision tree classifier
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

#make predictions
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred

#split original data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

#iterate over values of max depth ranging from 1-9, plot accuracy of models and training tests
# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

# Loop over different values of k
for i, k in enumerate(dep):
	# Setup a Decision Tree Classifier
	clf = tree.DecisionTreeClassifier(max_depth=k)

	# Fit the classifier to the training data
	clf.fit(X_train, y_train)

	#Compute accuracy on the training set
	train_accuracy[i] = clf.score(X_train, y_train)

	#Compute accuracy on the testing set
	test_accuracy[i] = clf.score(X_test, y_test)

# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


