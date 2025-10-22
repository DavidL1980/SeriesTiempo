import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print("="*60)
print("📊 PUNTO 2: LIMPIEZA DE DATOS - AAPL")
print("="*60)

# 1. CARGAR DATOS
print("\n1. CARGANDO DATOS...")
data = pd.read_csv('C:/Users/David/Documents/Perfil Upwork/Predicción de Series Temporales para Precios de Acciones (ML Avanzado)/aapl_data.csv', parse_dates=['Date'], index_col='Date')
print(f"   ✅ Shape original: {data.shape}")
print(f"   📋 Columnas: {list(data.columns)}")

# 2. DIAGNÓSTICO DE TIPOS
print("\n2. DIAGNÓSTICO DE TIPOS...")
print("   Tipos de datos:")
print(data.dtypes)
print(f"\n   Muestra de Close: {data['Close'].head(3).values}")

# 3. CONVERSIÓN A NUMÉRICO
print("\n3. CONVERSIÓN A NUMÉRICO...")
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
nan_count = data['Close'].isna().sum()
print(f"   ⚠️  NaN generados: {nan_count}")

# 4. LIMPIEZA
print("\n4. LIMPIEZA...")
data_clean = data.dropna(subset=['Close'])
print(f"   ✅ Shape después limpieza: {data_clean.shape}")
print(f"   ✅ Fechas desde: {data_clean.index.min()} hasta {data_clean.index.max()}")

# 5. OUTLIERS (IQR)
print("\n5. DETECCIÓN DE OUTLIERS...")
ts_data = data_clean['Close']
Q1 = ts_data.quantile(0.25)
Q3 = ts_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"   📊 Q1 (25%): ${Q1:.2f}")
print(f"   📊 Mediana: ${ts_data.median():.2f}")
print(f"   📊 Q3 (75%): ${Q3:.2f}")
print(f"   📊 IQR: ${IQR:.2f}")
print(f"   ⚠️  Límite inferior: ${lower_bound:.2f}")
print(f"   ⚠️  Límite superior: ${upper_bound:.2f}")

outliers_before = ((ts_data < lower_bound) | (ts_data > upper_bound)).sum()
print(f"   🚨 Outliers detectados: {outliers_before}")

# ELIMINAR OUTLIERS
ts_data_clean = ts_data[~((ts_data < lower_bound) | (ts_data > upper_bound))]
print(f"   ✅ Datos después outliers: {len(ts_data_clean)}")

# 6. RESAMPLE DIARIO
print("\n6. RESAMPLE DIARIO...")
ts_data_final = ts_data_clean.resample('D').ffill()
print(f"   ✅ Datos finales: {len(ts_data_final)} días")

# 7. ESTADÍSTICAS FINALES
print("\n7. ESTADÍSTICAS FINALES:")
stats = ts_data_final.describe()
print(stats.round(2))

# 8. GRÁFICO
plt.figure(figsize=(15, 8))
plt.subplot(2,1,1)
plt.plot(ts_data_final.index, ts_data_final.values, linewidth=0.8)
plt.title('AAPL - Precios de Cierre Limpios (2010-2025)', fontsize=14)
plt.ylabel('Precio ($)')
plt.grid(True, alpha=0.3)

plt.subplot(2,1,2)
plt.hist(ts_data_final, bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribución de Precios')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("✅ PUNTO 2 COMPLETADO - ¡COPIA TODA ESTA SALIDA!")
print("="*60)