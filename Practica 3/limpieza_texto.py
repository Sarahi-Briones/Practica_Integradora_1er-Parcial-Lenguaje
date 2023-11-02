import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  # Importa WordNetLemmatizer
import unicodedata
import string
from nltk.tokenize import word_tokenize
import re

# # Descargar la lista de stop words
# nltk.download('stopwords')

# # Descarga los recursos necesarios para la lematización en español
# nltk.download('omw')  # Diccionario WordNet en español

# Crea una instancia de WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def procesar_texto(texto):
    # Verifica si es una cadena de texto
    if isinstance(texto, str): 
        # Convierte el texto a minúsculas
        texto = texto.lower()

        # Elimina acentos y caracteres especiales
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'  or c == 'ñ')

        # Tokenizar el texto
        tokens = word_tokenize(texto, language='spanish')  # Tokenización del texto en español
        
        # Obtener los signos de puntuación
        puntuacion = set(string.punctuation)

        # Filtrar los tokens para eliminar la puntuación
        tokens = [token for token in tokens if token not in puntuacion]

        # Eliminar tokens que son números
        tokens = [token for token in tokens if not re.match(r'\d+', token)]

        # Elimina las stop words
        stop_words = set(stopwords.words('spanish'))  # Cambia 'spanish' al idioma de tu elección
        tokens = [word for word in tokens if word not in stop_words]

        # Lematiza los tokens
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Aplica lematización
        
        return ' '.join(tokens)
    else:
        return texto  # Si no es una cadena, devuelve el valor original
