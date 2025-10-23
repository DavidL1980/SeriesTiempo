import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, brier_score_loss
)

df = pd.read_csv("C:/Users/David/Documents/Cursos/Temario_especializacion_ciencia_de_datos/5-Machine Learning I (supervisado)/Data/pacientes.csv")
print(df.head())

print("\n ===================================================")
print(df.describe())

"""
NOTA: la DE es 1, no se muestra la 2DE y la 3DE 
con este resultado de estadísticas descriptivas podemos interpretar lo siguiente:
total de datos para las 3 variables: 300
edad:
media de 54.3, sugiere una población de media edad
DE de 9.09 indica que las edades varian moderadamente alrededor de la media. la mayoria de los pacientes tienen edades entre 45.2 y 63.4 años media+/- 1 DE
el mínimo es un paciente de 29 años
1er cuartil de 47, indica que el 25 de los pacientes tienen 47 años o menos
2do cuartil - la mediana, indica que la mitad de los pacientes so menores o iguales a 55 años. Es muy cerca de la media sugiriendo una distribución simetrica
El maximo es un paciente de 77 años

Colesterol:
media de 246.86 es el nivel promdeio de colesterol en los pacientes
DE de 51.56 muestra una variabilidad significativa 
minimo de 126 indica el nivel mínimo de colesterol
interpretación:
los niveles de colesterol son generalmente altos con una variable considerable, la precencia de un máxim de 564 sugier posibles outlier que podrian requerir
análisi adicional
"""

print("\n ===================================================")
print(df.info())

print("\n ================== separamos las variables ==========================")
y = df['problema_cardiaco']
X = df[['edad', 'colesterol']]
print(X.head())
print(y.head())

print("\n ================== Train/Test ==========================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#stratify=y: garantiza que la distribución de las clases en una varaible objetivo <y> se mantengan proporcionales en los conjuntos de entrenamiento y de prueba al dividir
# los datos

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n ================== Escalado de la variables independientes ==========================")
# lo que sigue se usa antes de entrenar el modelo, para asegurarse que las varaibles independientes estén en una escala comparable

scaler = StandardScaler()  # para crear una instancia o objeto
X_train_s = scaler.fit_transform(X_train)
"""
ajusta a <scaler> a los datos de entrenamiento y transforma X_train y test en una version de entrenamiento
fit calcula la media y la DE de cada columna en X_train y estos valores se almancenan en el objeto scaler
transform: apica la formula z = (x-u)/DE a cada valor en X_train generando una nueva matriz X_train_s. DOnde cada columna tienen la media = 0 y 
DE = 1
Sea plica a los datos de entrenamiento para evitar fugas de datos. Esto significa que el moddelo no "ve" información del conjunto de prueba X_test durante el entrenamiento
simulando un escenario real donde los datos de prueba son desconocidos
"""
X_test_s = scaler.transform(X_test)
"""
transforma el conjunto de datos X_test en una versión estandarizada utilizando la misma media y DE calculadas en X_train
transform: aplica la formula
no usa fit: para no recalcular la media y la DE
"""

print("\n ================== Entrenar la RLogistica ==========================")
# usamos  solver <liblinear> para datasets pequeños ajustando c para la regularización (1.0 default)
model = LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=1000)
"""
con la linea anterior se crea una instancia (objeto) con parámetros especificos para configurar el modelo. El proposito es definir el modelo que se usará para clasificar
si el paciente tienen o no el problema
solver='liblinear': el solver es el algoritmo utilizado para optimizar la funcion de pérdida de la RLog (minimizar el error). <liblinear> es unsolver eficiente para dataset
pequeños o medianos y es adecuado para problemas binarios
penalty='l2: agrega un termino de regularización a la función de pérdida para evitar el sobreajuste (overfitting), es decir los valores más grandes del modelo son penalizados
promoviendo un modelo más simple; el l2 (tambien llamdao ridge) suma el cuadrado de los ceoficientes al a funcion de perdida 
c = 1.0: C es el inverso de la fuerza de regularización y el 1.0 indica una regularizacion moderada
max_iter=1000:especifica el número máximo de iteraciones que solver puede realiar para converger a una solución optima 
"""
model.fit(X_train_s, y_train)
"""
entrena el modelo de RLog utilizando los datos de entrenamiento estandarizados y las etiquetas correspondientes a y_train
ajusta los coeficientes (betas) de la ecuación para minimizar la funcion de pérdida (log-loss) en los datos de entrenamiento
"""

print("\n ================== Predicciones y Probabilidades ==========================")

y_prob = model.predict_proba(X_test_s)[:,1]  # probabilidad de la clase 1
"""
esta linea usa el modelo de RLog para predecir las probabilidaes de que cada observación en X_test_s pertenezcan a la clase + (1)
predict_proba(X_test_s): calcula la probabilidad de cada clase para cada observación en X_test_s; para un problema binario devuelve una matriz en forma de [n_samples, 2], donde la 1ra
columna es la probabilidad de clase (- o 0); la 2da columna es la probilidad de la clase (+ o 1)
[:, 1]: selecciona solo la columna 1 de la matriz (probabilidades de 1)
"""
y_pred_05 = (y_prob >= 0.5).astype(int)  # threshold por defecto 0.5
"""
convierte las probabilidades en predicciones de clase binaria(0, 1) usando un umbral de 0.5
(y_prob >= 0.5): compara cada probabilida en y_prob con el umbral 0.5 devuelve un arreglo booleano (true o Flase)
True: si la probabilidad >= 0.5 (1)
Flase: sila probabilidad < 0.5 (0)
.astype(int): convierte los valores boolenaos en enteror (1/0)
"""

print("\n ================== Metricas ==========================")

acc = accuracy_score(y_test, y_pred_05)
prec = precision_score(y_test, y_pred_05)
rec = recall_score(y_test, y_pred_05)
f1 = f1_score(y_test, y_pred_05)
roc_auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\n=== Métricas (threshold=0.5) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(f"Brier score: {brier:.6f}")

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred_05, target_names= ["Sin Problema", "Con Problema"]))

print("\n ================== Matriz de confución ==========================")
cm = confusion_matrix(y_test, y_pred_05)
cm_df = pd.DataFrame(cm, index=['Actual_benign(0)','Actual_malignant(1)'],
                     columns=['Pred_benign(0)','Pred_malignant(1)'])
print("\n=== Matriz de confusión ===")
print(cm_df)

# Plot matriz
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.title("Matriz de confusión (threshold=0.5)")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

print("\n ================== ROC curve y AUC ==========================")

fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n ================== Precision-Recall curva ==========================")
precisions, recalls, pr_thresh = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(alpha=0.3)
plt.show()

print("\n ================== Efecto de múltiples thresholds ==========================")
thresholds = np.arange(0.1, 0.91, 0.1)
rows = []
for thr in thresholds:
    y_pred_thr = (y_prob >= thr).astype(int)
    rows.append({
        'threshold': thr,
        'accuracy': accuracy_score(y_test, y_pred_thr),
        'precision': precision_score(y_test, y_pred_thr, zero_division=0),
        'recall': recall_score(y_test, y_pred_thr, zero_division=0),
        'f1': f1_score(y_test, y_pred_thr, zero_division=0)
    })
thr_df = pd.DataFrame(rows)
print("\n=== Métricas para distintos thresholds ===")
print(thr_df)

# Mostrar tabla estilo
display(thr_df.style.format({
    'accuracy':'{:.4f}','precision':'{:.4f}','recall':'{:.4f}','f1':'{:.4f}'
}))

print("\n ================== Coeficientes y Odds Ratios ==========================")

coef = model.coef_.flatten()
intercept = model.intercept_[0]
coef_df = pd.DataFrame({'feature': X.columns, 'coef': coef})
coef_df['odds_ratio'] = np.exp(coef_df['coef'])
coef_df = coef_df.sort_values('odds_ratio', ascending=False).reset_index(drop=True)
print("\n=== Coeficientes (log-odds) y Odds Ratios (features estandarizadas) ===")
display(coef_df.head(15))
print(f"\nIntercept (log-odds): {intercept:.4f}")

# Interpretación ejemplo:
# - odds_ratio > 1 => aumento en la feature (1 std) multiplica las odds por ese factor.
# - odds_ratio < 1 => aumento en la feature disminuye las odds.

print("\n ================== VIF (multicolinealidad) - opcional ==========================")

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_train_vif = pd.DataFrame(X_train_s, columns=X.columns)
    vif = pd.DataFrame()
    vif['feature'] = X_train_vif.columns
    vif['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
    vif = vif.sort_values('VIF', ascending=False).reset_index(drop=True)
    print("\n=== VIF (top 15) ===")
    display(vif.head(15))
except Exception as e:
    print("\nVIF no pudo calcularse (statsmodels no disponible o error):", e)

print("\n ================== Calibración (Brier & reliability plot approximation) ==========================")
# Brier ya calculado. Podemos mostrar bins de probabilidad vs frecuencia observada.
df_cal = pd.DataFrame({'y': y_test, 'proba': y_prob})
df_cal['bin'] = pd.cut(df_cal['proba'], bins=np.linspace(0,1,6))
cal_table = df_cal.groupby('bin').agg(n=('y','size'), mean_proba=('proba','mean'), obs_rate=('y','mean')).reset_index()
print("\n=== Tabla de calibración (bins) ===")
display(cal_table)


print("\n ================== Small sample with predictions ==========================")
sample = X_test.reset_index(drop=True).loc[:9, :].copy()
sample['true'] = y_test.reset_index(drop=True).loc[:10]
sample['proba_malignant'] = y_prob[:10]
sample['pred_0.5'] = y_pred_05[:10]
print("\n=== Sample (primeras 10 filas del test) ===")
display(sample)

# ---------------------------
# FIN
# ---------------------------
print("\nEjercicio completado. Revisa las tablas y gráficas para interpretación.")