import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from fuzzywuzzy import fuzz

# wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=200)
# nlp = spacy.load("es_core_news_sm")

# df = pd.read_excel('archivo_procesado.xls')
# opiniones_hotel = df[df['Attraction'] == 'Hotel']['Opinion'].tolist()
# opiniones_restaurante = df[df['Attraction'] == 'Restaurant']['Opinion'].tolist()
# opiniones_atraccion = df[df['Attraction'] == 'Attractive']['Opinion'].tolist()
# text_hotel = ' '.join(opiniones_hotel)
# text_restaurante = ' '.join(opiniones_restaurante)
# text_atraccion = ' '.join(opiniones_atraccion)

# # Generamos el WordCloud de las opiniones de la categoría Hotel
# wordcloud.generate(text_hotel)
# # plt.figure(figsize=(10, 5))
# # plt.imshow(wordcloud, interpolation='bilinear')
# # plt.axis('off')  # Oculta los ejes
# # plt.show()
# # wordcloud.to_file('wordcloud_hotel.png')
# wordcloud_data_hotel = wordcloud.words_
# word_list_hotel = list(wordcloud_data_hotel.keys())
# doc_hotel = nlp(" ".join(word_list_hotel))
# etiquetas_pos_hotel = [(token.lemma_, token.pos_) for token in doc_hotel]
# adjetivos_hotel = []
# verbos_hotel = []
# sustantivos_hotel =[]
# ner_hotel =[]
# for palabra, etiqueta_pos in etiquetas_pos_hotel:
#     if etiqueta_pos=='ADJ':
#         adjetivos_hotel.append(palabra)
#     elif etiqueta_pos=='VERB':
#         verbos_hotel.append(palabra)
#     elif etiqueta_pos=='NOUN':
#         sustantivos_hotel.append(palabra)
#     elif etiqueta_pos=='PROPN':
#         ner_hotel.append(palabra)


# # Generamos el WordCloud de las opiniones de la categoría Restaurante
# wordcloud_restaurante= wordcloud.generate(text_restaurante)
# plt.figure(figsize=(10, 5))
# # plt.imshow(wordcloud, interpolation='bilinear')
# # plt.axis('off')  # Oculta los ejes
# # plt.show()
# # wordcloud.to_file('wordcloud_restaurante.png')
# wordcloud_data_restaurante = wordcloud.words_
# word_list_restaurante = list(wordcloud_data_restaurante.keys())
# doc_restaurante = nlp(" ".join(word_list_restaurante))
# etiquetas_pos_restaurante = [(token.lemma_, token.pos_) for token in doc_restaurante]
# adjetivos_restaurante = []
# verbos_restaurante = []
# sustantivos_restaurante =[]
# ner_restaurante =[]
# for palabra, etiqueta_pos in etiquetas_pos_restaurante:
#     if etiqueta_pos=='ADJ':
#         adjetivos_restaurante.append(palabra)
#     elif etiqueta_pos=='VERB':
#         verbos_restaurante.append(palabra)
#     elif etiqueta_pos=='NOUN':
#         sustantivos_restaurante.append(palabra)
#     elif etiqueta_pos=='PROPN':
#         ner_restaurante.append(palabra)
        

# # Generamos el WordCloud de las opiniones de la categoría Atracción
# wordcloud_atraccion= wordcloud.generate(text_atraccion)
# # plt.figure(figsize=(10, 5))
# # plt.imshow(wordcloud, interpolation='bilinear')
# # plt.axis('off')  # Oculta los ejes
# # plt.show()
# # wordcloud.to_file('wordcloud_atraccion.png')
# wordcloud_data_atraccion = wordcloud.words_
# word_list_atraccion = list(wordcloud_data_atraccion.keys())
# doc_atraccion = nlp(" ".join(word_list_atraccion))
# etiquetas_pos_atraccion = [(token.lemma_, token.pos_) for token in doc_atraccion]
# adjetivos_atraccion = []
# verbos_atraccion = []
# sustantivos_atraccion =[]
# ner_atraccion =[]
# for palabra, etiqueta_pos in etiquetas_pos_atraccion:
#     if etiqueta_pos=='ADJ':
#         adjetivos_atraccion.append(palabra)
#     elif etiqueta_pos=='VERB':
#         verbos_atraccion.append(palabra)
#     elif etiqueta_pos=='NOUN':
#         sustantivos_atraccion.append(palabra)
#     elif etiqueta_pos=='PROPN':
#         ner_atraccion.append(palabra)

# #----------------------Agregamos nuevas covariables el dataset------------------------
# columna_adjetivos_hotel=[]
# columna_verbos_hotel=[]
# columna_sustantivos_hotel=[]
# columna_ner_hotel=[]

# columna_adjetivos_restaurante=[]
# columna_verbos_restaurante=[]
# columna_sustantivos_restaurante=[]
# columna_ner_restaurante=[]

# columna_adjetivos_atraccion=[]
# columna_verbos_atraccion=[]
# columna_sustantivos_atraccion=[]
# columna_ner_atraccion=[]
# i=0
# for indice, fila in df.iterrows():
#     opinion = (fila['Opinion']).split()

#     contador_adjetivos_hotel = 0
#     contador_verbos_hotel = 0
#     contador_sustantivos_hotel = 0
#     contador_ner_hotel = 0

#     contador_adjetivos_restaurante = 0
#     contador_verbos_restaurante = 0
#     contador_sustantivos_restaurante = 0
#     contador_ner_restaurante = 0

#     contador_adjetivos_atraccion = 0
#     contador_verbos_atraccion = 0
#     contador_sustantivos_atraccion = 0
#     contador_ner_atraccion = 0

#     for token in opinion:
#         if str(token) in adjetivos_hotel:
#             contador_adjetivos_hotel+=1
#         else:
#             for palabra in adjetivos_hotel:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_adjetivos_hotel+=1
        
#         if str(token) in verbos_hotel:
#             contador_verbos_hotel+=1
#         else:
#             for palabra in verbos_hotel:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_verbos_hotel+=1
                    
#         if str(token) in sustantivos_hotel:
#             contador_sustantivos_hotel+=1
#         else:
#             for palabra in sustantivos_hotel:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_sustantivos_hotel+=1
                    
#         if str(token) in ner_hotel:
#             contador_ner_hotel+=1
#         else:
#             for palabra in ner_hotel:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_ner_hotel+=1

#         if str(token) in adjetivos_restaurante:
#             contador_adjetivos_restaurante+=1
#         else:
#             for palabra in adjetivos_restaurante:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_adjetivos_restaurante+=1
            
#         if str(token) in verbos_restaurante:
#             contador_verbos_restaurante+=1
#         else:
#             for palabra in verbos_restaurante:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_verbos_restaurante+=1
            
#         if str(token) in sustantivos_restaurante:
#             contador_sustantivos_restaurante+=1
#         else:
#             for palabra in sustantivos_restaurante:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_sustantivos_restaurante+=1 
            
#         if str(token) in ner_restaurante:
#             contador_ner_restaurante+=1
#         else:
#             for palabra in ner_restaurante:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_ner_restaurante+=1

#         if str(token) in adjetivos_atraccion:
#             contador_adjetivos_atraccion+=1
#         else:
#             for palabra in adjetivos_atraccion:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_adjetivos_atraccion+=1    
        
#         if str(token) in verbos_atraccion:
#             contador_verbos_atraccion+=1
#         else:
#             for palabra in verbos_atraccion:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_verbos_atraccion+=1
            
#         if str(token) in sustantivos_atraccion:
#             contador_sustantivos_atraccion+=1
#         else:
#             for palabra in sustantivos_atraccion:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_sustantivos_atraccion+=1
            
#         if str(token) in ner_atraccion:
#             contador_ner_atraccion+=1
#         else:
#             for palabra in ner_atraccion:
#                 sim = fuzz.ratio(str(token), palabra)
#                 if sim>=90:
#                     contador_ner_atraccion+=1
#     print(f"Opinion {i} recorrida")
#     i=i+1

#     columna_adjetivos_hotel.append(contador_adjetivos_hotel)
#     columna_verbos_hotel.append(contador_verbos_hotel)
#     columna_sustantivos_hotel.append(contador_sustantivos_hotel)
#     columna_ner_hotel.append(contador_ner_hotel)

#     columna_adjetivos_restaurante.append(contador_adjetivos_restaurante)
#     columna_verbos_restaurante.append(contador_verbos_restaurante)
#     columna_sustantivos_restaurante.append(contador_sustantivos_restaurante)
#     columna_ner_restaurante.append(contador_ner_restaurante)

#     columna_adjetivos_atraccion.append(contador_adjetivos_atraccion)
#     columna_verbos_atraccion.append(contador_verbos_atraccion)
#     columna_sustantivos_atraccion.append(contador_sustantivos_atraccion)
#     columna_ner_atraccion.append(contador_ner_atraccion)


# #--------------------------Nuevas covariables de hotel--------------------------------
# nuevo_df = df

# nuevo_df['Adjetivos_Hotel'] = columna_adjetivos_hotel
# nuevo_df['Verbos_Hotel'] = columna_verbos_hotel
# nuevo_df['Sustantivos_Hotel'] = columna_sustantivos_hotel
# nuevo_df['NER_Hotel'] = columna_ner_hotel
# #--------------------------Nuevas covariables de restaurante--------------------------
# nuevo_df['Adjetivos_Restaurante'] = columna_adjetivos_restaurante
# nuevo_df['Verbos_Restaurante'] = columna_verbos_hotel
# nuevo_df['Sustantivos_Restaurante'] = columna_sustantivos_restaurante
# nuevo_df['NER_Restaurante'] = columna_ner_restaurante
# #--------------------------Nuevas covariables de atraccion----------------------------
# nuevo_df['Adjetivos_Atraccion'] = columna_adjetivos_atraccion
# nuevo_df['Verbos_Atraccion'] = columna_verbos_atraccion
# nuevo_df['Sustantivos_Atraccion'] = columna_sustantivos_atraccion
# nuevo_df['NER_Atraccion'] = columna_ner_atraccion

# df.to_csv('dataset_parte_2.csv', index=False)

df = pd.read_csv('dataset_parte_2.csv')
x = df[['Adjetivos_Hotel','Verbos_Hotel','Sustantivos_Hotel','NER_Hotel','Adjetivos_Restaurante','Verbos_Restaurante','Sustantivos_Restaurante','NER_Restaurante','Adjetivos_Atraccion','Verbos_Atraccion','Sustantivos_Atraccion','NER_Atraccion']].values
y = df['Attraction'].values
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.0016,shuffle=True,random_state=0)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)
y_pred = naive_bayes.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", round(accuracy,4))

#Graficamos matriz de confusion
target_names = naive_bayes.classes_
cm = confusion_matrix(y_test, y_pred, labels=target_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot()
plt.show()