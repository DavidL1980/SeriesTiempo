import pandas as pd

#Cardar el data set
df = pd.read_csv('C:/Users/David/Documents/Cursos/Temario_especializacion_ciencia_de_datos/5-Machine Learning I (supervisado)/Data/train.csv')

print(df.head())
print("\n =========================================Información del dataset: \n")
print(df.info())
print("\n ================================================================== \n")

"""
Variables ppal:

1. survided: 0 = No, 1 = Si
2. pclass: Clase del boleto (1 = 1st, 2 = 2nd, 3 = 3rd)
3. sibsp: # de hermanos / esposos a bordo del titanic
4. parch: # de padres / hijos a bordo del titanic
5. fare: tarifa del pasajero
6. embarked: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

"""

# Revisar valores nulos
print(df.isnull().sum())
print("\n ================================================================== \n")

# imputar "Age" con la mediana
df['Age'] = df['Age'].fillna(df['Age'].median())  # reemplazar nulos con la mediana se usa para que no se vean afectados los outliers
# esto es aconsejable usar cuando hay muchos outliers, ya que la media se ve afectada por estos valores extremos.
# si los nulos son muchos (40% o mas) no es aconsejable imputar con la mediana (seria mucho ruido), es mejor eliminar la columna, ya que se perderia mucha informacion

"""
✅ Regla práctica:
Mediana → cuando hay outliers y la variable es numérica continua.
Media → cuando la distribución es simétrica y sin valores extremos.
Moda → para categóricos.
Modelos avanzados (KNN, regresión, ML) → cuando quieres aprovechar la relación entre variables para imputar.

"""

df ['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # reemplazar nulos con la moda (valor mas frecuente)

# dropear columnas pocos útiles o con muchos nulos
df = df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'])


#==============Codificacion de variables categóricas=================

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # codificación binaria

# one-hot encoding para embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # drop_first=True para evitar la trampa de la variable ficticia
# esto crea dos nuevas columnas: Embarked_Q y Embarked_S, con Embarked_C como la categoría base. Dicho de otra manera
# Convierte variables categóricas en variables binarias (0 o 1) para que los modelos de machine learning puedan procesarlas. esto se llama codificación one-hot.
# cada categoria que hay e la columna ['Embarked'] se convierte en una nueva columna con valores 0 o 1 indicando la presencia o ausencia de esa categoria.
"""
 drop_first=True
Elimina la primera columna dummy para evitar la trampa de la multicolinealidad (cuando una columna puede predecirse a partir de las demás en un modelo lineal).
En el ejemplo, en lugar de 3 columnas, se crean solo 2:
Embarked_C
Embarked_Q
Si ambas valen 0, significa que el pasajero estaba en S.
 
"""


#==============Escalado de variables numéricas=================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # creamos este objeto para escalar (normalizar) las variables numéricas
# z = (x - media) / desviación estándar
# esto centra los datos alrededor de 0 con una desviación estándar de 1, lo que ayuda a mejorar el rendimiento de muchos algoritmos de machine learning.
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
# fit: calcula la media y desviación estándar de las columnas 'Age' y 'Fare'.
# transform: aplica la transformación de escalado a estas columnas.


#==============Featuring Engineering=================
# se crean nuevas variables que pueden mejroar el modelo

# tamaño de la familia
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 para incluir al propio pasajero

# viaja solo
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)  # 1 si viaja solo, 0 si no

print(df.head())
print("\n ================================================================== \n")

#==================data set limpio para modelar==================
# separamos las varaibles predictoras (X) y la variable objetivo (y)
X = df.drop(columns=['Survived'])   # variables predictoras
y = df['Survived']                   # variable objetivo

print("Shape de X:", X.shape)
print("Shape de y:", y.shape)