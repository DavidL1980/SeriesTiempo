# GridSearch + SVM RBF (2D) con visualización de frontera y support vectors
# Requisitos: scikit-learn, matplotlib, numpy, pandas
# Ejecutar en Jupyter/Colab para ver gráficos en línea.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd

# ---------------------------
# 1) Preparar dataset 2D (no lineal) - make_circles
# ---------------------------
X, y = make_circles(n_samples=400, factor=0.5, noise=0.08, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Escalado (obligatorio para SVM)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------------------------
# 2) GridSearchCV: parámetros a explorar (C y gamma para RBF)
# ---------------------------
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 0.5, 1, 5, 'scale', 'auto']
}

# Stratified CV para mantener balance de clases por fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svc = SVC(kernel='rbf', probability=True, random_state=42)

grid = GridSearchCV(svc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train_s, y_train)

print("Best params (2D):", grid.best_params_)
print("Best CV AUC (2D):", grid.best_score_)

best_svc = grid.best_estimator_

# ---------------------------
# 3) Evaluar en test y mostrar métricas
# ---------------------------
y_pred = best_svc.predict(X_test_s)
y_proba = best_svc.predict_proba(X_test_s)[:,1]

print("\nTest accuracy (best SVM RBF):", accuracy_score(y_test, y_pred))
try:
    print("Test ROC AUC (best SVM RBF):", roc_auc_score(y_test, y_proba))
except Exception:
    pass
print("\nClassification report (test):\n", classification_report(y_test, y_pred))

# ---------------------------
# 4) Visualizar frontera de decisión y support vectors
# ---------------------------
def plot_decision_and_support(clf, X_all_s, y_all, title="Decision boundary"):
    # X_all_s: standardized points (can be train+test)
    x_min, x_max = X_all_s[:,0].min()-0.6, X_all_s[:,0].max()+0.6
    y_min, y_max = X_all_s[:,1].min()-0.6, X_all_s[:,1].max()+0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
    # plot points
    plt.scatter(X_all_s[:,0], X_all_s[:,1], c=y_all, s=30, edgecolor='k', cmap=plt.cm.Paired)
    # plot support vectors
    sv = clf.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], s=120, facecolors='none', edgecolors='r', linewidths=1.5, label='Support vectors')
    plt.legend()
    plt.xlabel("feature 1 (standardized)")
    plt.ylabel("feature 2 (standardized)")
    plt.title(title)
    plt.show()

# use train+test for visualization context
plot_decision_and_support(best_svc, np.vstack([X_train_s, X_test_s]), np.hstack([y_train, y_test]),
                          title=f"SVM RBF (best) — decision boundary (C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']})")

# ---------------------------
# 5) (Opcional) Visualizar la influencia de C/gamma en validación
# ---------------------------
# Construir DataFrame con resultados de GridSearch para inspección
cv_results = pd.DataFrame(grid.cv_results_)
# keep param, mean test score, std
inspect = cv_results[['param_C','param_gamma','mean_test_score','std_test_score']].sort_values('mean_test_score', ascending=False)
print("\nTop results from GridSearch (head):")
print(inspect.head(10).to_string(index=False))

# ---------------------------
# 6) (Opcional) GridSearch para Breast Cancer (high-dim) — mismo patrón
# ---------------------------
# Descomenta para ejecutar (puede tardar más)
"""
# data
data = load_breast_cancer(as_frame=True)
Xh = data.data
yh = data.target

# split + scale
Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, yh, test_size=0.25,
                                                       random_state=42, stratify=yh)
sc_h = StandardScaler().fit(Xh_train)
Xh_train_s = sc_h.transform(Xh_train)
Xh_test_s  = sc_h.transform(Xh_test)

# grid (tune fewer combos or use RandomizedSearchCV for speed)
param_grid_h = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 'scale']
}
svc_h = SVC(kernel='rbf', probability=True, random_state=42)
grid_h = GridSearchCV(svc_h, param_grid_h, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1, verbose=1)
grid_h.fit(Xh_train_s, yh_train)

print("Best params (breast):", grid_h.best_params_)
best_h = grid_h.best_estimator_
yh_pred = best_h.predict(Xh_test_s)
yh_proba = best_h.predict_proba(Xh_test_s)[:,1]
print("Test accuracy:", accuracy_score(yh_test, yh_pred))
print("Test ROC AUC:", roc_auc_score(yh_test, yh_proba))
print(classification_report(yh_test, yh_pred, target_names=data.target_names))
"""
# ---------------------------
# FIN
# ---------------------------

