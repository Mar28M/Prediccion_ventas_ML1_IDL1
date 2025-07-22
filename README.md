# Predicción de Ventas con Machine Learning_ML1_IDL1

Este proyecto implementa modelos de aprendizaje supervisado (clasificación y regresión) para predecir el comportamiento de clientes de una tienda virtual que vende productos electrónicos. Utiliza un conjunto de datos sintéticos generado especialmente para este caso práctico.

##  Objetivos

1. **Clasificación**: Predecir si un cliente realizará una compra, utilizando características demográficas y de comportamiento.
2. **Regresión**: Estimar el **monto total de la compra** en función de esas mismas variables.

##  Modelos Utilizados

###  Clasificación:
- **Árbol de Decisión (DecisionTreeClassifier)**

###  Regresión:
- **Regresión Lineal**
- **Regresión Polinómica**

##  Estructura del Proyecto

Prediccion_ventas_ML1_IDL1/
│
├── data_sintetica/                # Datos sintéticos (CSV)
├── notebooks/                     # Exploración y desarrollo de modelos en Jupyter
├── modelos_guardados/            # Modelos entrenados en formato .pkl
├── resultados_modelos/           # Predicciones y conjuntos de prueba exportados
├── scripts/                      # Scripts Python para entrenamiento
├── venv_IDL1_ML1_Mariana/        # Entorno virtual (ignorado por Git)
├── .gitignore
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Este archivo
└── informe_Mariana_Milagros.docx # Informe final del caso práctico (pendiente)

##  Cómo ejecutar
1. **Clonar el repositorio**

git clone https://github.com/Mar28M/Prediccion_ventas_ML1_IDL1.git
cd Prediccion_ventas_ML1_IDL1

2. **Crear y activar el entorno virtual**

python -m venv venv_IDL1_ML1_Mariana
venv_IDL1_ML1_Mariana\Scripts\activate.bat  # En Windows

3. **Instalar las dependencias**

pip install -r requirements.txt

4. **Ejecutar los scripts de entrenamiento**

python scripts\generar_datos_clientes.py
python scripts\entrenamiento_regresion_lineal.py
python scripts\entrenamiento_regresion_polinomica.py
python scripts\entrenamiento_clasificacion.py

También puedes usar los notebooks Jupyter para seguir el desarrollo paso a paso.

 **Visualizaciones generadas**
- Matriz de Confusión
- Curva ROC
- Gráficos de dispersión (predicciones vs valores reales)
- Importancia de características

**Librerías principales**
- scikit-learn, pandas, numpy, matplotlib, seaborn, joblib


**👩‍💻 Autora**
Mariana Milagros Inuma Macedo

Machine Learning I - Caso Practico Propuesto 1

📄 Informe final
El informe completo estará disponible en este repositorio.

