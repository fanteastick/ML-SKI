import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
boston=load_boston()


bos = pd.DataFrame(boston.data)

#name each column
bos.columns = boston.feature_names



bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis=1)

lm = LinearRegression()
print(lm)

lm.fit(X, bos.PRICE)



plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling(RM)")
plt.ylabel("Housing Price")
plt.title("relationship between RM and Price")
plt.show()

lm.predict(X)[0:5]

plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs predicted prices")

plt.show()