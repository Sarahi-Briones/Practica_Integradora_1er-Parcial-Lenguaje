import pandas as pd

# Crear un DataFrame de ejemplo
data = {'nombre': ['Alice', 'Bob', 'Charlie', 'David'],
        'opinion': ['buena', 'mala', None, 'excelente']}

df = pd.DataFrame(data)

# Eliminar filas con valores vacíos en la columna "opinion"
df = df.dropna(subset=['opinion']).reset_index(drop=True)

print(df)