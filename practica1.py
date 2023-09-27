import pandas as pd
import limpieza_texto
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#-----------------------------------------------------------Limpieza de dataset---------------------------------------------------- 
# df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')
# #borramos opiniones vacias
# df = df.dropna(subset=['Opinion']).reset_index(drop=True)
# # Aplica la función a la columna
# df['Opinion'] = df['Opinion'].apply(limpieza_texto.procesar_texto)
# df.to_excel('archivo_procesado.xls', index=False, engine='openpyxl')

#----------------------------------------------------Extramos opiniones y atracciones del dataset----------------------------------
# df = pd.read_excel('archivo_procesado.xls')
# opiniones = df['Opinion']
# atracciones = df['Attraction']

#------------------------------------------------------Definimos variables para el modelo-----------------------------------------
# lista_opiniones = opiniones.values.tolist()
# lista_atracciones = atracciones.values.tolist()
# tagged_data = [TaggedDocument(word_tokenize(_o),_a) for _o,_a in zip(lista_opiniones,lista_atracciones)]
# model = Doc2Vec(vector_size=300, window=8, min_count=5, workers=4, alpha=0.025, min_alpha=0.025, dm = 0, dbow_words=1)
# model.build_vocab(tagged_data)

#--------------------------------------------------------------Entrenamos el modelo-----------------------------------------------
# print("Empieza el entrenamiento")
# max_epochs = 20
# i=0
# for epoch in range(max_epochs):
#     model.train(tagged_data,total_examples = model.corpus_count, epochs = model.epochs)
#     model.alpha =-0.0002
#     model.min_alpha = model.alpha
#     print(f"epoca: {i}")
#     i=i+1
# print("Fin del entrenamiento")
# model.save("d2v_01.model")

#---------------------------------------------------------Creacion de vectores con Doc2vec-----------------------------------------
#Abrimos el dataset limpio para asignar las etiquetas a los vectores de caracteristicas de las opiniones
modelo_doc2vec = Doc2Vec.load("d2v_01.model")
df = pd.read_excel('archivo_procesado.xls')
opiniones = df['Opinion']
atracciones = df['Attraction']
lista_opiniones = opiniones.values.tolist()
lista_atracciones = atracciones.values.tolist()

# Crear vectores de características para las opiniones
vectores_de_caracteristicas = [modelo_doc2vec.infer_vector(doc.lower().split()) for doc in lista_opiniones]

#------------------------------------------------------Entrenamiento modelo KNN-----------------------------------------------------
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(vectores_de_caracteristicas, lista_atracciones, test_size=0.25, random_state=42)

# Entrenar el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predecir las etiquetas para los datos de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión
precision = accuracy_score(y_test, y_pred)
print("Precisión del clasificador KNN:", precision)