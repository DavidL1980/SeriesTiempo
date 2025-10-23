import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


## cargar los datos
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# escalado, es estrictamente necesario para árboles, pero útil si combinas con otros modelos
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# entrenar el modelo
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train_s, y_train)

# Prediccion y metricas
y_pred = reg.predict(X_test_s)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Métricas (árbol sin podar) ===")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# 6) Visualizar árbol (solo para árboles pequeños; limitar max_depth para que sea legible)
plt.figure(figsize=(16,8))
plot_tree(reg, feature_names=X.columns, filled=True, max_depth=3, fontsize=10)
plt.title("Decision Tree Regressor (primeros 3 niveles)")
plt.show()

# 7) Importancia de features
feat_imp = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", feat_imp.head(10))

# 8) Diagnóstico: real vs predicho y residuales
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Valores reales vs predichos (árbol regresión)")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_pred, y_test - y_pred, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicho")
plt.ylabel("Residual")
plt.title("Residuales vs Predichos")
plt.show()
# 9) Poda por cost-complexity (CCP) - buscar ccp_alpha óptimo usando validación
path = reg.cost_complexity_pruning_path(X_train_s, y_train)
ccp_alphas = path.ccp_alphas
# Para evitar centenares de árboles, muestreamos algunos al final del rango
ccp_alphas = ccp_alphas[::max(1, len(ccp_alphas)//20)]

param_grid = {'ccp_alpha': ccp_alphas}
grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train_s, y_train)
best_alpha = grid.best_params_['ccp_alpha']
print("\nMejor ccp_alpha (GridSearchCV):", best_alpha)

# Entrenar árbol podado con alpha óptimo
reg_pruned = DecisionTreeRegressor(random_state=42, ccp_alpha=best_alpha)
reg_pruned.fit(X_train_s, y_train)
y_pred_p = reg_pruned.predict(X_test_s)

mse_p = mean_squared_error(y_test, y_pred_p)
rmse_p = np.sqrt(mse_p)
r2_p = r2_score(y_test, y_pred_p)
print(f"Podded tree -> RMSE: {rmse_p:.4f}, R^2: {r2_p:.4f}")

# Comparar tamaño original vs podado (profundidad)
print("Profundidad (original):", reg.get_depth(), " | hojas:", reg.get_n_leaves())
print("Profundidad (podado):", reg_pruned.get_depth(), " | hojas:", reg_pruned.get_n_leaves)