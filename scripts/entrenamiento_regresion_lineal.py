# 1. Importar librerías
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 2. Cargar y preparar datos
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_path = os.path.join(base_dir, "data_sintetica", "clientes_tienda_virtual.csv")

df = pd.read_csv(data_path)
df = df[df['total_compra'] > 0].copy()

X = df[['grupo_edad', 'region', 'nivel_ingresos', 'horas_conectado', 'clics_en_productos', 'uso_cupones']]
y = df['total_compra']

cat_cols = ['grupo_edad', 'region', 'nivel_ingresos']

# 3. Definir pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])

# 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento
pipeline.fit(X_train, y_train)

# 6. Evaluación sobre test
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f" Evaluación en test:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 7. Validación cruzada
scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
y_cv_pred = cross_val_predict(pipeline, X, y, cv=5)
cv_rmse = np.sqrt(mean_squared_error(y, y_cv_pred))

print("\n Validación cruzada:")
print("R² por fold:", np.round(scores, 3))
print(f"R² promedio (CV): {scores.mean():.3f}")
print(f"RMSE promedio (CV): {cv_rmse:.2f}")

# 8. Guardar modelo
modelo_path = os.path.join(base_dir, "modelos_guardados", "modelo_regresion_lineal.pkl")
os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
joblib.dump(pipeline, modelo_path)

# 9. Guardar predicciones
output_dir = os.path.join(base_dir, "resultados_modelos", "regresion")
os.makedirs(output_dir, exist_ok=True)

resultados = pd.DataFrame({
    'real': y_test,
    'predicho': y_pred
})
pred_path = os.path.join(output_dir, "prediccion_regresion_lineal.csv")
resultados.to_csv(pred_path, index=False, encoding='utf-8-sig')

# 10. Guardar conjuntos de prueba
X_test.to_csv(os.path.join(output_dir, "X_test_erl.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test_erl.csv"), index=False)

# 11. Confirmación final
print("\n Modelo y predicciones guardados correctamente.")
print(f" Modelo: {modelo_path}")
print(f" Predicciones: {pred_path}")
