import pandas as pd
import limpieza_texto

df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
#borramos opiniones vacias
df = df.dropna(subset=['Opinion']).reset_index(drop=True)
# Aplica la función a la columna
df['Opinion'] = df['Opinion'].apply(limpieza_texto.procesar_texto)
df.to_excel('archivo_procesado.xls', index=False, engine='openpyxl')
print(df['Opinion'])