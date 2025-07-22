# 1. Importar librerías
import pandas as pd
import numpy as np
import os
import random

# Para reproducibilidad
np.random.seed(42)

# 2. Definir parámetros
n_clientes = 15000

grupos_edad = ['18-25', '26-35', '36-45', '46-60', '60+']
niveles_ingresos = ['bajo', 'medio', 'alto']

# Lista de todas las regiones del Perú
regiones = [
    'Amazonas', 'Áncash', 'Apurímac', 'Arequipa', 'Ayacucho', 'Cajamarca',
    'Callao', 'Cusco', 'Huancavelica', 'Huánuco', 'Ica', 'Junín',
    'La Libertad', 'Lambayeque', 'Lima', 'Loreto', 'Madre de Dios',
    'Moquegua', 'Pasco', 'Piura', 'Puno', 'San Martín', 'Tacna',
    'Tumbes', 'Ucayali'
]

# Asignar probabilidad 25% a Lima, el resto se reparte equitativamente
n_regiones = len(regiones)
probs = [0.0] * n_regiones
lima_index = regiones.index('Lima')

probs[lima_index] = 0.25
valor_para_otros = (1 - 0.25) / (n_regiones - 1)

for i in range(n_regiones):
    if i != lima_index:
        probs[i] = valor_para_otros

# Verificación opcional
print("Probabilidades por región (suma total):", round(sum(probs), 4))

# 3. Generar datos
data = {
    'ID_Cliente': [f"CL{str(i).zfill(5)}" for i in range(1, n_clientes + 1)],
    'grupo_edad': np.random.choice(grupos_edad, n_clientes),
    'region': np.random.choice(regiones, n_clientes, p=probs),
    'nivel_ingresos': np.random.choice(niveles_ingresos, n_clientes, p=[0.3, 0.5, 0.2]),
    'horas_conectado': np.round(np.random.normal(loc=2.5, scale=1.0, size=n_clientes), 2),
    'clics_en_productos': np.random.poisson(lam=5, size=n_clientes),
    'uso_cupones': np.random.randint(0, 5, size=n_clientes),
}

df = pd.DataFrame(data)

# Corregir posibles valores negativos
df['horas_conectado'] = df['horas_conectado'].clip(lower=0)

# 4. Simular compra
def probabilidad_compra(row):
    base = 0.2
    if row['nivel_ingresos'] == 'medio':
        base += 0.1
    elif row['nivel_ingresos'] == 'alto':
        base += 0.2
    base += 0.05 * min(row['uso_cupones'], 3)
    base += 0.02 * min(row['clics_en_productos'], 10)
    return min(base, 0.95)

df['prob_compra'] = df.apply(probabilidad_compra, axis=1)
df['compra_realizada'] = np.random.binomial(1, df['prob_compra'])

# 5. Generar total de compra (solo si compró)
def generar_monto(compra, ingreso):
    if compra == 0:
        return 0.0
    base = np.random.uniform(100, 1000)
    if ingreso == 'alto':
        base *= 2
    elif ingreso == 'medio':
        base *= 1.3
    return round(base, 2)

df['total_compra'] = df.apply(lambda row: generar_monto(row['compra_realizada'], row['nivel_ingresos']), axis=1)

# Eliminar columna auxiliar
df.drop(columns=['prob_compra'], inplace=True)

# 6. Guardar como CSV
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, 'data_sintetica')

df.to_csv(os.path.join(output_dir, 'clientes_tienda_virtual.csv'), index=False, encoding='utf-8-sig')

print(" Dataset generado correctamente:", os.path.join(output_dir, 'clientes_tienda_virtual.csv'))