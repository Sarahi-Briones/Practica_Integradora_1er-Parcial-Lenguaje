import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis',  # Cambia el colormap según tus preferencias de color
    max_words=200,       # Número máximo de palabras a mostrar en el Word Cloud
    font_path='ruta/a/tu/fuente.ttf'  # Ruta a una fuente personalizada (opcional)
)

