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
data_path = os.path.join("..", "data_sintetica", "clientes_tienda_virtual.csv")
df = pd.read_csv(data_path)

df = df[df['total_compra'] > 0].copy()

X = df[['grupo_edad', 'region', 'nivel_ingresos', 'horas_conectado', 'clics_en_productos', 'uso_cupones']]
y = df['total_compra']

cat_cols = ['grupo_edad', 'region', 'nivel_ingresos']

# 3. Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regression', LinearRegression())
])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento
pipeline.fit(X_train, y_train)

# 6. Evaluación
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 7. Validación cruzada
scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print("R² por fold:", np.round(scores, 3))
print("R² promedio (CV):", scores.mean())

# RMSE con predicción cruzada
y_cv_pred = cross_val_predict(pipeline, X, y, cv=5)
cv_rmse = np.sqrt(mean_squared_error(y, y_cv_pred))
print(f"RMSE promedio (CV): {cv_rmse:.2f}")

# 8. Guardar modelo
modelo_path = os.path.join("..", "modelos_guardados", "modelo_regresion_lineal.pkl")
joblib.dump(pipeline, modelo_path)

# 9. Guardar predicciones
resultados = pd.DataFrame({
    'real': y_test,
    'predicho': y_pred
})
output_path = os.path.join("..", "resultados_modelos", "regresion", "prediccion_regresion_lineal.csv")
resultados.to_csv(output_path, index=False, encoding='utf-8-sig')

# 10. Guardar test
X_test.to_csv(os.path.join("..", "resultados_modelos", "regresion", "X_test_erl.csv"), index=False)
y_test.to_csv(os.path.join("..", "resultados_modelos", "regresion", "y_test_erl.csv"), index=False)
print(" Modelo y predicciones guardados.")