# Predicción de ventas comerciales mediante Machine Learning — Rossmann Store Sales

**Proyecto final de Machine Learning | Maestría en Ciencia de Datos**

**Autor:** César Toapanta  
**Año:** 2026  
**Herramientas:** Python, Google Colab, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib y Seaborn.

---

## 1. Descripción del proyecto

Este proyecto desarrolla y compara modelos de aprendizaje automático para predecir las ventas diarias de los establecimientos de la cadena Rossmann, utilizando información histórica, variables temporales y características comerciales y operativas.

El estudio se basa en el conjunto de datos **Rossmann Store Sales**, disponible en Kaggle, y busca identificar un modelo predictivo que pueda contribuir a la planificación de demanda, la gestión de inventarios y la toma de decisiones comerciales.

Se comparan dos algoritmos de regresión:

- **Random Forest Regressor**
- **XGBoost Regressor**

La metodología comprende análisis exploratorio de datos (EDA), ingeniería de características, selección de variables, construcción de pipelines, optimización de hiperparámetros, validación temporal, comparación estadística, curva de aprendizaje y visualización mediante t-SNE.

## 2. Objetivos

### Objetivo general

Desarrollar y comparar modelos de Machine Learning para estimar las ventas diarias de los establecimientos Rossmann, evaluando su desempeño predictivo sobre períodos futuros.

### Objetivos específicos

1. Analizar la calidad, distribución y comportamiento temporal de los datos.
2. Preparar las variables predictoras mediante técnicas de imputación y codificación.
3. Aplicar selección de características mediante el método filter `SelectKBest`.
4. Implementar pipelines reproducibles utilizando Scikit-learn.
5. Optimizar los hiperparámetros de Random Forest y XGBoost.
6. Evaluar los modelos mediante validación cronológica y validación cruzada repetida.
7. Comparar estadísticamente sus errores de predicción.
8. Identificar aplicaciones potenciales para la planificación comercial.

## 3. Conjunto de datos

**Fuente:** [Rossmann Store Sales — Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales)

El conjunto de datos contiene información de ventas de 1.115 establecimientos comerciales durante el período comprendido entre el 1 de enero de 2013 y el 31 de julio de 2015.

| Característica | Descripción |
|---|---|
| Observaciones originales | 1.017.209 |
| Establecimientos | 1.115 |
| Período | Enero de 2013 a julio de 2015 |
| Variable objetivo | `Sales` |
| Variables predictoras iniciales | 18 |
| Observaciones utilizadas para modelado | 844.338 |
| Observaciones de entrenamiento | 785.727 |
| Observaciones de prueba | 58.611 |

Para el modelado se consideraron únicamente observaciones correspondientes a tiendas abiertas con ventas positivas.

Se excluyó la variable `Customers` para evitar utilizar información que no estaría disponible antes del día de predicción.

## 4. Estructura del repositorio

```text
Proyecto-ML-chtoapantam-MSDS/
│
├── README.md
├── Rossmann_Proyecto_ML_chtoapantam.ipynb
├── train.parquet
├── store.csv
├── test.csv
└── sample_submission.csv
```

**Descripción de archivos:**

- `Rossmann_Proyecto_ML_chtoapantam.ipynb`: notebook principal con el análisis, los modelos, las visualizaciones y las conclusiones.
- `train.parquet`: datos históricos de entrenamiento almacenados en formato Parquet.
- `store.csv`: características de los establecimientos.
- `test.csv`: datos adicionales proporcionados por la competencia.
- `sample_submission.csv`: estructura de ejemplo para las predicciones de Kaggle.
- `README.md`: documentación del proyecto.

## 5. Metodología

### 5.1. Análisis exploratorio de datos (EDA)

Se realizó un análisis descriptivo y gráfico para identificar:

- Distribución de las ventas.
- Diferencias de ventas según día de la semana y mes.
- Relación descriptiva entre promociones y ventas.
- Comportamiento de ventas según tipo de establecimiento.
- Valores faltantes y posibles anomalías.
- Evolución temporal de las ventas.

### 5.2. Ingeniería y selección de características

Se incorporaron variables derivadas de las fechas:

- Año.
- Mes.
- Semana del año.
- Día del mes.

Se implementaron transformaciones mediante `ColumnTransformer` y `Pipeline`, incluyendo:

- `SimpleImputer` para valores faltantes.
- `OneHotEncoder` para variables categóricas.
- `SelectKBest` con `f_regression` para selección de características.

La selección de características se integró dentro de los pipelines para evitar ajustar transformaciones con información de validación.

### 5.3. Modelos utilizados

**Random Forest Regressor:** algoritmo basado en un conjunto de árboles de decisión, que permite representar relaciones no lineales entre las variables predictoras y las ventas.

**XGBoost Regressor:** algoritmo de boosting basado en árboles de decisión, que construye modelos secuenciales para reducir los errores de predicción.

### 5.4. Optimización de hiperparámetros

Se utilizó `RandomizedSearchCV` con cuatro particiones temporales, evaluando ocho configuraciones por algoritmo.

Para reducir el costo computacional, la búsqueda se realizó sobre una muestra reproducible de 156.837 observaciones del conjunto de entrenamiento.

| Modelo | Mejor RMSE promedio de validación |
|---|---:|
| Random Forest | 1.607,80 |
| **XGBoost** | **1.589,80** |

### 5.5. Evaluación cronológica

Se reservó un conjunto de prueba correspondiente a los últimos dos meses del período disponible.

Esta estrategia permite evaluar la capacidad predictiva de los modelos sobre observaciones posteriores a las utilizadas durante el entrenamiento.

### 5.6. Comparación estadística

Se realizaron evaluaciones complementarias utilizando:

- Repeated K-Fold con 5 particiones y 2 repeticiones.
- Prueba t corregida para validación cruzada repetida.
- Comparación exploratoria de errores semanales mediante Wilcoxon.

También se construyó una curva de aprendizaje y se aplicó t-SNE para visualizar una representación bidimensional del espacio de características.

## 6. Resultados principales

### 6.1. Evaluación sobre el conjunto de prueba cronológico

| Métrica | Random Forest | XGBoost |
|---|---:|---:|
| **RMSE** | **1.318,24** | 1.484,95 |
| **MAE** | **903,19** | 1.094,21 |
| **R²** | **0,8205** | 0,7722 |

Random Forest obtuvo mejores resultados en las tres métricas evaluadas.

En particular, presentó un RMSE aproximadamente un 11,23 % menor que XGBoost.

### 6.2. Validación cruzada repetida

| Indicador | Random Forest | XGBoost |
|---|---:|---:|
| RMSE promedio | 1.370,45 | **1.360,37** |
| Desviación estándar | **17,09** | 20,08 |
| Particiones con menor RMSE | 2 | **8** |

XGBoost obtuvo una ligera ventaja en el RMSE promedio de la validación cruzada repetida.

Sin embargo, la prueba t corregida produjo los siguientes resultados:

| Indicador estadístico | Resultado |
|---|---:|
| Estadístico t | 1,1357 |
| P-valor | 0,285430 |
| Intervalo de confianza del 95 % | [-9,99; 30,14] |
| Nivel de significancia | 0,05 |

Dado que el p-valor fue superior a 0,05, no se rechazó la hipótesis nula. Por tanto, no se encontró evidencia estadística suficiente para concluir que existe una diferencia sistemática entre los errores medios de ambos algoritmos bajo este procedimiento.

La validación cruzada repetida se considera un análisis complementario, puesto que no preserva el orden cronológico de las observaciones.

## 7. Modelo seleccionado

**Modelo candidato principal: Random Forest Regressor.**

La selección se fundamenta en su mejor desempeño sobre el conjunto de prueba cronológico:

- RMSE: 1.318,24.
- MAE: 903,19.
- R²: 0,8205.

Dado que el objetivo consiste en pronosticar ventas futuras, se prioriza la evaluación temporal frente a las particiones aleatorias de Repeated K-Fold.

La selección corresponde al período y las condiciones evaluadas, por lo que se recomienda realizar validaciones adicionales antes de una implementación empresarial.

## 8. Cómo ejecutar el proyecto

### Opción 1. Google Colab

**[Abrir notebook en Google Colab](https://colab.research.google.com/drive/1uTTI7J4pX9nUq_hqscwmxAl3jAzjCZQW?usp=sharing)**

1. Abrir el enlace de Google Colab.
2. Seleccionar un entorno de ejecución de Python.
3. Instalar las dependencias necesarias si el entorno lo requiere.
4. Ejecutar las celdas en orden, desde la carga de datos hasta las conclusiones.

El notebook carga los archivos principales desde el repositorio público de GitHub.

### Opción 2. Desde GitHub

Clonar el repositorio:

```bash
git clone https://github.com/ctoa99/Proyecto-ML-chtoapantam-MSDS.git
```

Ingresar al directorio:

```bash
cd Proyecto-ML-chtoapantam-MSDS
```

Instalar las bibliotecas principales:

```bash
pip install pandas numpy pyarrow scikit-learn xgboost matplotlib seaborn scipy jupyter
```

Abrir el notebook:

```bash
jupyter notebook Rossmann_Proyecto_ML_chtoapantam.ipynb
```

**Nota:** las búsquedas de hiperparámetros y las evaluaciones repetidas pueden requerir varios minutos de ejecución. La duración depende de los recursos computacionales disponibles.

## 9. Aplicaciones comerciales

Los resultados del proyecto permiten identificar posibles aplicaciones del Machine Learning en procesos de gestión comercial:

- **Planificación de demanda:** anticipar ventas por tienda y período.
- **Gestión de inventarios:** apoyar decisiones de abastecimiento y reposición.
- **Promociones comerciales:** estimar la demanda esperada considerando variables promocionales.
- **Negociaciones con proveedores:** utilizar pronósticos como insumo para planificar volúmenes de compra.
- **Planificación operativa:** facilitar la asignación de recursos según la demanda prevista.

Estas aplicaciones son potenciales y requieren validación adicional antes de su utilización operativa.

## 10. Limitaciones

El estudio presenta algunas limitaciones:

- Los datos corresponden al período 2013-2015.
- El análisis se centra en una única cadena comercial.
- Repeated K-Fold no conserva la estructura temporal.
- La comparación estadística repetida utiliza hiperparámetros seleccionados previamente.
- No se incorporaron variables externas como indicadores económicos o condiciones meteorológicas.
- La evaluación se centró en tiendas abiertas con ventas positivas.

Como trabajo futuro, se propone incorporar rezagos de ventas, promedios móviles, variables externas y nuevas estrategias de validación temporal.

## 11. Conclusiones

El proyecto permitió desarrollar una metodología integral de Machine Learning para predecir ventas comerciales.

Los algoritmos Random Forest y XGBoost mostraron resultados competitivos, aunque sus desempeños relativos variaron según la estrategia de evaluación.

XGBoost obtuvo una ligera ventaja en la validación cruzada repetida; sin embargo, la prueba t corregida no identificó evidencia suficiente de una diferencia estadísticamente significativa al nivel del 5 %.

Random Forest presentó el mejor desempeño en la evaluación cronológica, con un RMSE de 1.318,24 y un R² de 0,8205.

Por tanto, **Random Forest se seleccionó como el modelo candidato principal para el pronóstico de ventas**, destacando la importancia de utilizar estrategias de validación coherentes con la naturaleza temporal del problema.

## 12. Referencias principales

[1] Kaggle, "Rossmann Store Sales," 2015. Disponible: https://www.kaggle.com/competitions/rossmann-store-sales

[2] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.

[3] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proceedings of ACM SIGKDD*, pp. 785–794, 2016.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[5] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.

[6] C. Nadeau and Y. Bengio, "Inference for the Generalization Error," *Machine Learning*, vol. 52, pp. 239–281, 2003.

## 13. Uso de inteligencia artificial generativa

Durante el desarrollo del proyecto se utilizó ChatGPT (OpenAI) como herramienta de apoyo para la explicación de conceptos, estructuración del código, revisión de errores e interpretación de resultados.

Los experimentos y las métricas presentadas corresponden a las ejecuciones realizadas en Google Colab.

---

**Proyecto académico — Maestría en Ciencia de Datos — 2026**
