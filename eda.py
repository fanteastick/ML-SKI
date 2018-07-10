import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

sns.set()

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train.head(n=4))
print (df_train.info())
print (df_train.describe())

sns.countplot(x='Survived', data=df_train)


df_test['Survived'] = 0
print(df_train.groupby(['Sex']).Survived.sum())


print ("proportion of women and men that survived")
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()


df_train.groupby('Survived').Fare.hist(alpha=0.6);