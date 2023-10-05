import nltk
from pattern.es import parse, split

# Descargar el recurso necesario para NLTK
nltk.download('punkt')

# Texto de ejemplo
texto = "El rápido zorro marrón saltó sobre el perro perezoso."

# Tokenizar y etiquetar palabras en español
sentences = split(parse(texto, tokenize=True, tags=True, chunks=False))
adjetivos = []

# Extraer adjetivos etiquetados como 'JJ' (adjetivos en inglés) o 'JJR' (comparativo de adjetivos en inglés)
for sentence in sentences:
    for word, tag in sentence:
        if tag in ['JJ', 'JJR']:
            adjetivos.append(word)

# Imprimir los adjetivos
print(adjetivos)