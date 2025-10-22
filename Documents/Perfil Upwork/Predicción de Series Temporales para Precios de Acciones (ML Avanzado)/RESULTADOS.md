# **RESULTADOS**



El análisis de estacionariedad de los precios de cierre de AAPL (2010-2025) confirma que la serie original NO es estacionaria (p-value = 0.995), mostrando una clara tendencia alcista de $6 a $262 con crecimiento exponencial. La prueba ADF rechaza la estacionariedad debido a la media cambiante a lo largo del tiempo, visualizada en el gráfico como una rampa ascendente con picos notables en 2020 (COVID) y 2024 (IA).



La primera diferenciación (ΔClose) transforma exitosamente la serie en estacionaria (p-value = 4.18e-24), eliminando la tendencia y estabilizando la media alrededor de $0 con oscilaciones diarias de ±20$. El gráfico diferenciada muestra cambios diarios sin dirección sistemática, confirmando que d=1 es suficiente para ARIMA. Esto permite modelar directamente los retornos diarios y reconstruir precios futuros sumando predicciones acumuladas.





El análisis comparativo de los tres modelos (ARIMA, Prophet, LSTM) sobre precios de AAPL (2022-2025) muestra que LSTM captura mejor la volatilidad reciente, siguiendo de cerca los picos y valles reales ($180-$260), mientras ARIMA subestima movimientos extremos y Prophet ofrece una tendencia suave pero conservadora. En el período de prueba, LSTM mantiene la menor desviación respecto a los valores reales, demostrando superioridad en patrones no lineales cortoplacistas.



El forecast futuro (2025-2026) proyecta un crecimiento moderado con LSTM alcanzando $280 (escenario optimista), ARIMA estabilizándose en $220 (conservador) y Prophet en $240 (intermedio). Todos los modelos coinciden en tendencia alcista sostenida, aunque LSTM muestra mayor variabilidad realista, reflejando la volatilidad histórica de AAPL.



