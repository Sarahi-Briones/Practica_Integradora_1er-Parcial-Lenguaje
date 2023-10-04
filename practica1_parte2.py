import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis',  # Cambia el colormap según tus preferencias de color
    max_words=200       # Número máximo de palabras a mostrar en el Word Cloud
)

df = pd.read_excel('archivo_procesado.xls')

opiniones_hotel = df[df['Attraction'] == 'Hotel']['Opinion'].tolist()
opiniones_restaurante = df[df['Attraction'] == 'Restaurant']['Opinion'].tolist()
opiniones_atraccion = df[df['Attraction'] == 'Attractive']['Opinion'].tolist()
text_hotel = ' '.join(opiniones_hotel)
text_restaurante = ' '.join(opiniones_restaurante)
text_atraccion = ' '.join(opiniones_atraccion)

# Generamos el WordCloud de las opiniones de la categoría Hotel
wordcloud.generate(text_hotel)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Oculta los ejes
plt.show()
wordcloud.to_file('wordcloud_hotel.png')

# Generamos el WordCloud de las opiniones de la categoría Restaurante
wordcloud_restaurante= wordcloud.generate(text_restaurante)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Oculta los ejes
plt.show()
wordcloud.to_file('wordcloud_restaurante.png')

# Generamos el WordCloud de las opiniones de la categoría Atracción
wordcloud_atraccion= wordcloud.generate(text_atraccion)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Oculta los ejes
plt.show()
wordcloud.to_file('wordcloud_atraccion.png')
