# 1. Importar librerías
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 2. Rutas base
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_path = os.path.join(base_dir, "data_sintetica", "clientes_tienda_virtual.csv")

# 3. Cargar datos
df = pd.read_csv(data_path)
df = df[df['total_compra'] > 0].copy()

X = df[['grupo_edad', 'region', 'nivel_ingresos', 'horas_conectado', 'clics_en_productos', 'uso_cupones']]
y = df['total_compra']

cat_cols = ['grupo_edad', 'region', 'nivel_ingresos']
num_cols = ['horas_conectado', 'clics_en_productos', 'uso_cupones']

# 4. Pipeline con polinomios
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols),
    ('poly', PolynomialFeatures(degree=2, include_bias=False), num_cols)
])

pipeline = Pipeline([
    ('transform', preprocessor),
    ('regression', LinearRegression())
])

# 5. Split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 6. Evaluación
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
modelo_path = os.path.join(base_dir, "modelos_guardados", "modelo_regresion_polinomica.pkl")
os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
joblib.dump(pipeline, modelo_path)

# 9. Guardar predicciones
output_dir = os.path.join(base_dir, "resultados_modelos", "regresion")
os.makedirs(output_dir, exist_ok=True)

resultados = pd.DataFrame({
    'real': y_test,
    'predicho': y_pred
})
pred_path = os.path.join(output_dir, "prediccion_regresion_polinomica.csv")
resultados.to_csv(pred_path, index=False, encoding='utf-8-sig')

# 10. Guardar test
X_test.to_csv(os.path.join(output_dir, "X_test_poly.csv"), index=False)
y_test.to_csv(os.path.join(output_dir," y_test_poly.csv"), index=False)

# 11. Confirmación final
print("\n Modelo polinómico y resultados guardados correctamente.")
print(f" Modelo: {modelo_path}")
print(f" Predicciones: {pred_path}")
