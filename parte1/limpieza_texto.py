import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  # Importa WordNetLemmatizer
import unicodedata
import re

# # Descargar la lista de stop words
# nltk.download('stopwords')

# # Descarga los recursos necesarios para la lematización en español
# nltk.download('wordnet')
# nltk.download('omw')  # Diccionario WordNet en español

# Crea una instancia de WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def procesar_texto(texto):
    if isinstance(texto, str):  # Verifica si es una cadena de texto
        # Convierte el texto a minúsculas
        texto = texto.lower()
        
        # Elimina acentos y caracteres especiales
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'  or c == 'ñ')

        # Tokeniza el texto
        tokens = texto.split()

        # Elimina las stop words y signos de puntuacion
        stop_words = set(stopwords.words('spanish'))  # Cambia 'spanish' al idioma de tu elección
        tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens if word not in stop_words]

        # Lematiza los tokens
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Aplica lematización

        return ' '.join(tokens)
    else:
        return texto  # Si no es una cadena, devuelve el valor original
    
# df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
# #borramos opiniones vacias
# df = df.dropna(subset=['Opinion']).reset_index(drop=True)
# # Aplica la función a la columna
# df['Opinion'] = df['Opinion'].apply(procesar_texto)
# df.to_excel('archivo_procesado.xls', index=False, engine='openpyxl')