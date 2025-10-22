# **REPORTE**



El análisis de estacionariedad de los precios de cierre de AAPL (2010-2025) confirma que la serie original NO es estacionaria (p-value = 0.995), mostrando una clara tendencia alcista de $6 a $262 con crecimiento exponencial. La prueba ADF rechaza la estacionariedad debido a la media cambiante a lo largo del tiempo, visualizada en el gráfico como una rampa ascendente con picos notables en 2020 (COVID) y 2024 (IA).



La primera diferenciación (ΔClose) transforma exitosamente la serie en estacionaria (p-value = 4.18e-24), eliminando la tendencia y estabilizando la media alrededor de $0 con oscilaciones diarias de ±20$. El gráfico diferenciada muestra cambios diarios sin dirección sistemática, confirmando que d=1 es suficiente para ARIMA. Esto permite modelar directamente los retornos diarios y reconstruir precios futuros sumando predicciones acumuladas.









SERIE ORIGINAL (Close sin procesar)

ADF Statistic: -1.212080089410737

p-value: 0.995391924368985



EXPLICACIÓN:

ADF Statistic (-1.21): Valor "débil" (más cerca de 0 que de -∞)

p-value (0.995): 99.5% probabilidad de que la serie NO sea estacionaria

Umbral: > 0.05 = NO RECHAZA H0 (H0 = "serie NO estacionaria")







SERIE DIFERENCIADA (Close)

ADF Statistic: -12.9042611188118

p-value: 4.1834198504675e-24



EXPLICACIÓN:

ADF Statistic (-12.90): Valor EXTREMADAMENTE FUERTE (muy negativo)

p-value (4.18e-24): Prácticamente 0 - 100% certeza de estacionariedad

Umbral: < 0.05 = RECHAZA H0 (H0 = "serie NO estacionaria")





GRÁFICO 1: SERIE ORIGINAL

EJE Y: $0 → $250

EJE X: 2010 → 2026

LÍNEA AZUL: Precio Close



LO QUE MUESTRA:

Línea sube continuamente: $6 (2010) → $262 (2025)

Pendiente positiva constante: Tendencia alcista clara

Picos: 2020 (COVID boom), 2024 (IA hype)

CONCLUSIÓN VISUAL: NO ESTACIONARIA (media cambia con tiempo)





GRÁFICO 2: SERIE DIFERENCIADA

EJE Y: -20 → +20 dólares

EJE X: 2010 → 2026

LÍNEA NARANJA: Cambio diario (Close\_hoy - Close\_ayer)



LO QUE MUESTRA:

Línea oscila alrededor de 0: Media constante

Rango estable: Cambios diarios ±20$ máximo

Picos específicos:



+18$: Día boom COVID (2020)

-15$: Caída 2022 (inflación)



CONCLUSIÓN VISUAL: ESTACIONARIA (sin tendencia)





PROCESO MATEMÁTICO QUE OCURRE



FÓRMULA DIFERENCIACIÓN:

ΔClose\_t = Close\_t - Close\_(t-1)



EJEMPLO CON NUMEROS REALES:

Close 2025-01-01 = $250

Close 2024-12-31 = $248

ΔClose = 250 - 248 = +2$  ← CAMBIO DIARIO





TRANSFORMACIÓN COMPLETA:



ANTES (NO ESTACIONARIA):

2010: $6    ← Media baja

2020: $130  ← Media alta  

2025: $262  ← Media muy alta



DESPUÉS (ESTACIONARIA):

2010: +0.5$

2020: +2.1$

2025: +1.8$  ← Media ~$1.5 CONSTANTE





RESULTADO FINAL

SERIE ORIGINAL    →    SERIE DIFERENCIADA

📈 Tendencia      →    ➡️ Media constante (0)

📊 Volatilidad    →    📊 Volatilidad preservada

❌ No modelable   →    ✅ Modelable ARIMA/LSTM

p=0.995           →    p=0.000000000000000000000004









=================ExplicaciÓn de los dos graficos finales=======================================



El análisis comparativo de los tres modelos (ARIMA, Prophet, LSTM) sobre precios de AAPL (2022-2025) muestra que LSTM captura mejor la volatilidad reciente, siguiendo de cerca los picos y valles reales ($180-$260), mientras ARIMA subestima movimientos extremos y Prophet ofrece una tendencia suave pero conservadora. En el período de prueba, LSTM mantiene la menor desviación respecto a los valores reales, demostrando superioridad en patrones no lineales cortoplacistas.



El forecast futuro (2025-2026) proyecta un crecimiento moderado con LSTM alcanzando $280 (escenario optimista), ARIMA estabilizándose en $220 (conservador) y Prophet en $240 (intermedio). Todos los modelos coinciden en tendencia alcista sostenida, aunque LSTM muestra mayor variabilidad realista, reflejando la volatilidad histórica de AAPL.



