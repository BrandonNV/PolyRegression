import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
wine = pd.read_csv('./winequality-red.csv')
print(wine.columns)
print(wine.info())
X = wine[['quality']]
y = wine[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = PolynomialFeatures(degree= 4)
y_ = model.fit_transform(y)
y_test_ = model.fit_transform(y_test)
lg = LinearRegression()
lg.fit(y_,X)
predicted_data = lg.predict(y_test_)
predicted_data = np.round_(predicted_data)
print ('Error',mean_squared_error(X_test,predicted_data))
print (predicted_data)



