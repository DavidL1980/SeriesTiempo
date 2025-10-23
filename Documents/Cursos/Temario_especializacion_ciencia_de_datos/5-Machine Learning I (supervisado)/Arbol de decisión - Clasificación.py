# Árbol de decisión - Clasificación (Breast Cancer)
# Librerías: scikit-learn, pandas, numpy, matplotlib, seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# 1) Cargar datos
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 1 = malignant, 0 = benign

# 2) Train/test split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Escalado (no obligatorio para árboles)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4) Entrenar árbol clasificador (control de complejidad: max_depth)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_s, y_train)

# 5) Predicción y métricas
y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:,1]  # prob de clase positiva

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("=== Métricas (árbol por defecto) ===")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 6) Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred benign (0)','Pred malig (1)'],
            yticklabels=['True benign (0)','True malig (1)'])
plt.title("Confusion matrix")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# 7) ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 8) Importancia de features y visualización del árbol (primeros niveles)
feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", feat_imp.head(10))

plt.figure(figsize=(16,8))
plot_tree(clf, feature_names=X.columns, class_names=data.target_names, filled=True, max_depth=3, fontsize=10)
plt.title("Decision Tree Classifier (primeros 3 niveles)")
plt.show()

# 9) Regularización por grid search (control max_depth, min_samples_leaf)
param_grid = {
    'max_depth': [2, 3, 4, 5, 8, None],
    'min_samples_leaf': [1, 2, 5, 10]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring='f1')
grid.fit(X_train_s, y_train)
print("\nMejor params (GridSearch):", grid.best_params_, "Mejor score (CV f1):", grid.best_score_)

best_clf = grid.best_estimator_
y_pred_best = best_clf.predict(X_test_s)
y_proba_best = best_clf.predict_proba(X_test_s)[:,1]
print("\n=== Métricas (árbol podado por GridSearch) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Precision:", precision_score(y_test, y_pred_best))
print("Recall:", recall_score(y_test, y_pred_best))
print("F1:", f1_score(y_test, y_pred_best))
print("ROC AUC:", roc_auc_score(y_test, y_proba_best))

# 10) Comparar tamaños
print("Profundidad (default):", clf.get_depth(), " | hojas:", clf.get_n_leaves())
print("Profundidad (best):", best_clf.get_depth(), " | hojas:", best_clf.get_n_leaves())
