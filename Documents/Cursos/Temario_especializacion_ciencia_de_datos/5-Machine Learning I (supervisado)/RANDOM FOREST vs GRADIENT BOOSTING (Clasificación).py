# RANDOM FOREST vs GRADIENT BOOSTING (Clasificación)
# Dataset: Breast Cancer (sklearn)
# Requisitos: scikit-learn, pandas, numpy, matplotlib, seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# 1) Datos
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
print("Shape:", X.shape, "Positives ratio:", y.mean())

# 2) Train/test (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) (Opcional) Escalado: para árboles no es obligatorio, pero para comparar con otros modelos puede usarse
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4) Modelos: default + un GB con parámetros típicos
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

# 5) Entrenamiento
rf.fit(X_train, y_train)      # Random forest suele funcionar bien sin escalado
gb.fit(X_train, y_train)      # GradientBoosting tampoco necesita escalado, uso raw X

# 6) Predicciones y probabilidades
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:,1]

# 7) Métricas (function)
def print_metrics(y_true, y_pred, y_proba, name="model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC AUC: {auc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=data.target_names))

print_metrics(y_test, y_pred_rf, y_proba_rf, "Random Forest")
print_metrics(y_test, y_pred_gb, y_proba_gb, "Gradient Boosting")

# 8) Matrices de confusión
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()

plot_cm(y_test, y_pred_rf, "Confusion Matrix - Random Forest")
plot_cm(y_test, y_pred_gb, "Confusion Matrix - Gradient Boosting")

# 9) Curvas ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)
plt.figure(figsize=(6,5))
plt.plot(fpr_rf, tpr_rf, label=f'RF AUC={roc_auc_score(y_test,y_proba_rf):.4f}')
plt.plot(fpr_gb, tpr_gb, label=f'GB AUC={roc_auc_score(y_test,y_proba_gb):.4f}')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 10) Importancias de features (top 10)
def plot_feature_importances(model, X, title, top_n=10):
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:top_n]
    imp.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()
    
plot_feature_importances(rf, X, "RF feature importances (top10)")
plot_feature_importances(gb, X, "GB feature importances (top10)")

# 11) Cross-validation estimate (stratified)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_auc = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
gb_cv_auc = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print("CV ROC AUC (RF):", rf_cv_auc.mean(), "+/-", rf_cv_auc.std())
print("CV ROC AUC (GB):", gb_cv_auc.mean(), "+/-", gb_cv_auc.std())

# 12) Comentarios rápidos:
# - Compara precision/recall si priorizas FP vs FN.
# - Recomendación: RandomForest = robusto y fácil de usar;
#   GradientBoosting = suele alcanzar mayor precisión si afinado (learning_rate, n_estimators, max_depth).
