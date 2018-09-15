import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Lecutra de datos
'''
El objetivo del ejercicio es tratar de predecir la calidad del vino tomando
en cuenta los factores que se nos dan en el dataset utilizando la tecnica de
Regresion Polinomial
'''
#Lectura de datos
wine = pd.read_csv('./winequality-red.csv')
#Impreison de los headings de cada columna
print(wine.columns)
#Impresion para saber si hay valores null dentro del dataframe
#En este caso todas las columnas nos dieron como resultdao non-null y de tipo de dato todas son float64, excepto quality que es int64
print(wine.info())
#Información de cuartiles, minimo, máximo y promedio para determinar valores fuera de rango.
print(wine.describe())
#Variable a ser predecida
Y = wine[['quality']]
#Seleccion de variables basada en factores más relevantes
x = wine[['density','fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
#Las variables de prueba son del 20%
X_train, X_test, y_train, y_test = train_test_split(Y, x, test_size=0.2)
'''
El modelo se establecio con caracteristicas polinomiales, es decir que
genere una nueva matriz de funciones que consiste en todas las 
combinaciones polinomiales de las características con un 
grado menor o igual al grado especificado
Niveles elevados pueden causar overfitting
'''
#Se probaron grados 1,2,3,4,4 y el 4 resulto el resultado más adecuado
model = PolynomialFeatures(degree= 4)
x_ = model.fit_transform(x)
y_test_ = model.fit_transform(y_test)
#Regresion lineal con sckit
lg = LinearRegression()
lg.fit(x_,Y)
#Prediccion de la calidad del vino
predicted_data = lg.predict(y_test_)
predicted_data = np.round_(predicted_data)
#Error
print ('Mean squared error',mean_squared_error(X_test,predicted_data))
# Valor de R en donde la predicción con valor 1 es perfect
print('Variance score: %.2f' % r2_score(X_test, predicted_data))
