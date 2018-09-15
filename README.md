# Linear regression with polinomial features

Un patrón común en el aprendizaje automático es usar modelos lineales entrenados 
en funciones no lineales de los datos. Esto permite extender los modelos lineales y
adaptarse a una gama de datos mucho más ampllia manteniendo el mismo rendimiento

En este caso se utiizo un dataset para predecir la calidad del vino, el ejercicio
tuvo como resultado que el coeficiente R fue de .95 es decir muy cercano  a 1
por lo que se realizo una predicción mucho más certera a diferencia de que se hubiera
realizado con regresión lineal simple


Los factores que se decidieron tomar para realizar el ejercicio fueron los siguientes:
{
'density','fixed acidity','volatile acidity','citric acid','
residual sugar','chlorides','free sulfur dioxide',
'total sulfur dioxide','density','pH','sulphates','alcohol'
}

Problema y dataset obtenido de:
https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
