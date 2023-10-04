import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import spacy

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=200)
nlp = spacy.load("en_core_web_sm")

df = pd.read_excel('archivo_procesado.xls')
opiniones_hotel = df[df['Attraction'] == 'Hotel']['Opinion'].tolist()
opiniones_restaurante = df[df['Attraction'] == 'Restaurant']['Opinion'].tolist()
opiniones_atraccion = df[df['Attraction'] == 'Attractive']['Opinion'].tolist()
text_hotel = ' '.join(opiniones_hotel)
text_restaurante = ' '.join(opiniones_restaurante)
text_atraccion = ' '.join(opiniones_atraccion)

# Generamos el WordCloud de las opiniones de la categoría Hotel
wordcloud.generate(text_hotel)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Oculta los ejes
# plt.show()
# wordcloud.to_file('wordcloud_hotel.png')
wordcloud_data_hotel = wordcloud.words_
word_list_hotel = list(wordcloud_data_hotel.keys())
#frequencies_hotel = list(wordcloud_data_hotel.values())
doc_hotel = nlp(" ".join(word_list_hotel))
etiquetas_pos_hotel = [(token.text, token.pos_) for token in doc_hotel]

# Generamos el WordCloud de las opiniones de la categoría Restaurante
wordcloud_restaurante= wordcloud.generate(text_restaurante)
plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Oculta los ejes
# plt.show()
# wordcloud.to_file('wordcloud_restaurante.png')
wordcloud_data_restaurante = wordcloud.words_
word_list_restaurante = list(wordcloud_data_restaurante.keys())
#frequencies_restaurante = list(wordcloud_data_restaurante.values())
doc_restaurante = nlp(" ".join(word_list_restaurante))
etiquetas_pos_restaurante = [(token.text, token.pos_) for token in doc_restaurante]

# Generamos el WordCloud de las opiniones de la categoría Atracción
wordcloud_atraccion= wordcloud.generate(text_atraccion)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Oculta los ejes
# plt.show()
# wordcloud.to_file('wordcloud_atraccion.png')
wordcloud_data_atraccion = wordcloud.words_
word_list_atraccion = list(wordcloud_data_atraccion.keys())
#frequencies_atraccion = list(wordcloud_data_atraccion.values())
doc_atraccion = nlp(" ".join(word_list_atraccion))
etiquetas_pos_atraccion = [(token.text, token.pos_) for token in doc_atraccion]

print(etiquetas_pos_hotel)