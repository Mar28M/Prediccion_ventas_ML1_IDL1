# 1. Importar librerías
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 2. Definir rutas absolutas
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_path = os.path.join(base_dir, "data_sintetica", "clientes_tienda_virtual.csv")

# 3. Cargar datos
df = pd.read_csv(data_path)

X = df[['grupo_edad', 'region', 'nivel_ingresos', 'horas_conectado', 'clics_en_productos', 'uso_cupones']]
y = df['compra_realizada']

cat_cols = ['grupo_edad', 'region', 'nivel_ingresos']

# 4. Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

# 5. División y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 6. Evaluación
y_pred = pipeline.predict(X_test)
print(" Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 7. Validación cruzada
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("\n Validación cruzada:")
print("Accuracy por fold:", np.round(scores, 3))
print(f"Accuracy promedio (CV): {scores.mean():.2f}")

# 8. Guardar modelo
modelo_path = os.path.join(base_dir, "modelos_guardados", "arbol_decision_modelo.pkl")
os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
joblib.dump(pipeline, modelo_path)

# 9. Guardar predicciones
output_dir = os.path.join(base_dir, "resultados_modelos", "clasificacion")
os.makedirs(output_dir, exist_ok=True)

resultados = pd.DataFrame({
    'real': y_test,
    'predicho': y_pred
})
pred_path = os.path.join(output_dir, "prediccion_arbol_de_decisión.csv")
resultados.to_csv(pred_path, index=False, encoding='utf-8-sig')

# 10. Guardar X_test e y_test
X_test.to_csv(os.path.join(output_dir, "X_test_clasificacion.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test_clasificacion.csv"), index=False)

# 11. Confirmación
print("\n Modelo de clasificación y resultados guardados correctamente.")
print(f" Modelo: {modelo_path}")
print(f" Predicciones: {pred_path}")