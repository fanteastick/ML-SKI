#importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

sns.set()

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#store target variable of training data in safe place
survived_train = df_train.Survived

#concatenate training n test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# Extract Title from Name, store in column and plot barplot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);


#view a barplot of all the titles
data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);
#plt.show()

#making a 'has cabin' feature
data['Has_Cabin']= ~data.Cabin.isnull()

#drop columns
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)

#fill unfinished columns with na
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna('S')
print (data.info())

#bin the numerical data
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False)
data['CatFare'] = pd.qcut(data.Fare, q=4, labels=False)

#drop the data you just binned
data=data.drop(['Age','Fare'], axis=1)

#column of family members on board
data['Fam_Size'] = data.Parch + data.SibSp
data= data.drop(['SibSp', 'Parch'], axis=1)

#transform everything to binary variables
#data_dum = pd.get_dummies(data, drop_first = True)
#print(data_dum.head())





