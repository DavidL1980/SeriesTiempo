# RANDOM FOREST vs GRADIENT BOOSTING (Regresión)
# Dataset: California Housing
# Requisitos: scikit-learn, pandas, numpy, matplotlib, seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1) Datos
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
print("Shape:", X.shape)

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3) Modelos
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
gb_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)

# 4) Entrenar
rf_reg.fit(X_train, y_train)
gb_reg.fit(X_train, y_train)

# 5) Predicción y métricas
def regression_metrics(y_true, y_pred, label="model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

y_pred_rf = rf_reg.predict(X_test)
y_pred_gb = gb_reg.predict(X_test)

regression_metrics(y_test, y_pred_rf, "Random Forest Regressor")
regression_metrics(y_test, y_pred_gb, "Gradient Boosting Regressor")

# 6) Residuales vs predichos
plt.figure(figsize=(6,4))
plt.scatter(y_pred_rf, y_test - y_pred_rf, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicho (RF)")
plt.ylabel("Residual")
plt.title("Residuales vs Predichos (RF)")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_pred_gb, y_test - y_pred_gb, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicho (GB)")
plt.ylabel("Residual")
plt.title("Residuales vs Predichos (GB)")
plt.show()

# 7) Importancia de features
def plot_feat_reg(model, X, title):
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:12]
    imp.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()

plot_feat_reg(rf_reg, X, "RF regressor feature importances (top12)")
plot_feat_reg(gb_reg, X, "GB regressor feature importances (top12)")

# 8) Cross-validation RMSE (KFold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_mse = cross_val_score(rf_reg, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
gb_cv_mse = cross_val_score(gb_reg, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
rf_cv_rmse = np.sqrt(-rf_cv_mse)
gb_cv_rmse = np.sqrt(-gb_cv_mse)
print("CV RMSE (RF):", rf_cv_rmse.mean(), "+/-", rf_cv_rmse.std())
print("CV RMSE (GB):", gb_cv_rmse.mean(), "+/-", gb_cv_rmse.std())

# 9) Comentarios rápidos:
# - Compara RMSE con la std o media de y para juzgar si el error es aceptable.
# - Random Forest suele ser más robusto; Gradient Boosting puede ser más preciso si se afinan hyperparámetros.
