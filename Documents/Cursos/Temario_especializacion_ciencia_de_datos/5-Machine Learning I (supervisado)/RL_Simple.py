import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# carga del data set
df = pd.read_csv('C:/Users/David/Documents/Cursos/Temario_especializacion_ciencia_de_datos/5-Machine Learning I (supervisado)/Data/clean_cafe_saless.csv')
print(df.head())
print("\n =========================================Información del dataset: \n")
print(df.info())
"""
Con la información del dataset podemos ver que no hay valores nulos y que todas las variables son numéricas (int64 o float64).
"""
print("\n ================================================================== \n")
print(df.describe())
"""
Los resultados obtenidos con la descripción estadística del dataset nos permiten observar lo siguiente:
1. Quantity: la cantidad de ítems pos trx esta limitada a un rango discreto de (1 a 5) loq ue sugiere que el dataset podría estar restringido a un máximo de 5 ítem
por trx.
La distribucion parece simetrica o ligeramente sesgada, ya que la media de 2.98 y la media de 3 son muy cercanas
La desviacion estandar (DE) de 1.37 indica que las cantidades no varían mucho, con la mayoria de las trx concentrdas entre 2 y 4 ítems (según los percentiles)

2. Price Per Unit: los precios unitarios están restringidos aun rango discreto (1 a 5), lo que sugiere que los ítem tienen precios fijos discretos entre $1 y $5
La media de 2.9 y la mediana de 3 son muy cercanas, lo que indica una distribución aproximadamente simétrica
La DE de 1.23 muestra que los precios unitarios no varían mucho con la mayoria de los ítems teniendo precios entre $2 y $4 8 según percentiles)
"""

print("\n ================================================================== \n")
# seleccion de las variables

# dependiente 
y = df['Total Spent']

# independiente
X = df[['Price Per Unit', 'Quantity']]

"""
Si X solo fuera una variable X = df['Total Spent'], esto seria solo una serie de 1D tocaría pasarlo a una matris 2D de la sguiente manera
X_train = df['columna'].values.reshape(-1, 1)  # Convierte a un arreglo 2D de forma (n_samples, 1)
otra manera sería:
X_train = df['columna'].to_numpy().reshape(-1, 1)
"""

# división en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crear el modelo
modelo = LinearRegression()

# entrenar el modelo
modelo.fit(X_train, y_train)

# predicciones
predicciones = modelo.predict(X_test)

# coeficiente y bias
print("Intercepto (B_0): ", modelo.intercept_)
print("coeficiente (B_1, B_2): ", modelo.coef_)

"""
Intercepto (B_0):  -7.370395082403217. Esto representa el valor estimado de la variable "y" cuando todas las variables independientes "X's" son = 0.
cuando las X's = 0 el modelo predice que total gastado sería aproximadamente de -7.37 (unidad monetaria). Un intercepto negativo no tiene sentido en este caso, ya que no es
posible tener un gasto total negativo ni un precio unitario = 0.

coeficiente (B_1, B_2):  [2.70676794 2.67745112].  El B_1 indica que por cada unidad adicional en <Quantity> ( es decir que por cada ítem adicional comprado) el total_spend
aumenta en promdeio 2.71 unidades monetarias asumiendo que el precio el Price Per Unit se mantiene constante. Esto implica que si el precio unitario no cambia, comprar un
ítem aumenta el gasto total en aproximadamente $2.71, este valor es razonable, ya que el precio  unitario promedio en el dataset es aproximadamente $2.9 y el coeficiente
esta cercano a este

EN RESUMEN
Para este caso se esperaria que el B_0 fuera 0 y que los coeficientes fueran exactamente proporcionales a las variables, sin embargo:
El valor negativo del intercepto, sugiere que el modelo no captura perfectamente la relacion multiplicativa exacta
"""
print("\n ================================================================== \n")

# evaluacion del modelo

#Calculo de métricas
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

"""
MSE: 5.839957932416181. El cuadrado de los errores. El MSE penaliza más los errores grandes (debido al termino cuadratico) este valor sugiere que el modelo
tiene algunos errores significativos, pero para interpretarlo mejor, necesitamos compararlo con la escala de <y>, como esta variable tienen un rango de 1 a 20
y una media de 8.44, un MSE de 5.84 e smoderado en este contexto pero indica que las predicciones no son perfectas

RMSE: 2.416600490858218. Indica que las desviaciones se desvian este valor de los valores reales de <y> dado que la media de <y> fue de 8.44 y DE = 5.33 un 
RMSE de 2.42 representa un error relativo moderado (aproximadamente 28.7% de la media 2.42/8.44 = 0.287), esto sugiere que el modelo tiene un ajuste 
razonable

MAE: 1.6798718482182142. este valor indica que, en promedio, las predicciones del modelo se desvian 1.68 unidades monetarias de los valores reales de <y>
no penaliza los errores grandes. Un MAE mas bajo que el RMSE sugiere que los errores grandes son relativamente pocos, pero existen, ya que el RMSE es más
sensibles que estos

R2: 0.7881629404391716. este valor indica que el 78,8% de la variabilidad en <y> es explicada por las variables independientes <X's>, esto sugiere un ajuste
razonablemente bueno. El 21.2% restante de la variabilidad no esta explicada por el modelo lo que podria deberse a la relacion multiplicativa no capturada
por el modelo, posible ruido en los datos o las varaibles omitidas
"""

print("\n ================================================================== \n")

# Visualización de los datos

# comparar valores reales vs predicos
plt.figure(figsize=(6, 6))
sns.scatterplot(x = y_test, y = predicciones, alpha= 0.5)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title(" Regresión lineal: valroes reales vs predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = "red", linestyle = "--")
plt.show()