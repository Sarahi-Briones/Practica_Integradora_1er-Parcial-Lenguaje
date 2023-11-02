import pandas as pd
import limpieza_texto
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pickle
from sklearn.metrics import silhouette_score
#-----------------------------------------------------------Limpieza de dataset---------------------------------------------------- 
# df = pd.read_excel('Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx')

# # Borramos opiniones vacias
# df = df.dropna(subset=['Opinion']).reset_index(drop=True)

# # Aplica la función a la columna
# df['Opinion'] = df['Opinion'].apply(limpieza_texto.procesar_texto)

# # Guardar nuevo dataset
# df.to_excel('archivo_procesado.xls', index=False, engine='openpyxl')

#----------------------------------------------------Extramos opiniones y atracciones del dataset----------------------------------
df = pd.read_excel('archivo_procesado.xls')
opiniones = df['Opinion']
atracciones = df['Attraction']

#------------------------------------------------------Definimos variables para el modelo-----------------------------------------
lista_opiniones = opiniones.values.tolist()
lista_atracciones = atracciones.values.tolist()
#tagged_data = [TaggedDocument(word_tokenize(_o),word_tokenize(_a)) for _o,_a in zip(lista_opiniones,lista_atracciones)]


# vector_size_list = [50,100,150]
# window_list = [5,10,15]

# best_model = None
# best_accuracy = 0

# # Realizar la búsqueda de hiperparámetros
# for vector_size in vector_size_list:
#     for window in window_list:
#         # Entrenar modelo Doc2Vec con combinación de hiperparámetros actual
#         model = Doc2Vec(vector_size=vector_size, window=window, min_count=1, workers=4)
#         model.build_vocab(tagged_data)
#         model.train(tagged_data, total_examples=model.corpus_count, epochs=10)

#         # Evaluación del modelo
#         #test_targets, test_regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in test_docs])
        
#         vectores_de_caracteristicas = [model.infer_vector(doc.lower().split()) for doc in lista_opiniones]

#         X_train, X_test, y_train, y_test = train_test_split(vectores_de_caracteristicas, lista_atracciones, test_size=0.2, shuffle=True)

#         # Entrenar el clasificador KNN
#         knn = KNeighborsClassifier(n_neighbors=3)
#         knn.fit(X_train, y_train)

#         # Predecir las etiquetas para los datos de prueba
#         y_pred = knn.predict(X_test)

#         accuracy = accuracy_score(y_test, y_pred)

#         # Actualizar el mejor modelo si se mejora la precisión
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model
#             best_hiperparametros = [vector_size,window]
#         print(accuracy)

# # El mejor modelo Doc2Vec y su configuración de hiperparámetros
# print("Mejor precisión:", best_accuracy)
# print("Configuración de hiperparámetros:", best_hiperparametros)
# best_model.save("d2v_01.model")


#NOTA: los mejores hiperparamteros fueron: vector size= 50, window = 5



#------------------------------Entrenar modelo Doc2Vec con la mejor combinacion de hiperparametros-----------------------------------------------
# model = Doc2Vec(vector_size=50, window=5, min_count=1, workers=4)
# model.build_vocab(tagged_data)
# model.train(tagged_data, total_examples=model.corpus_count, epochs=500) #200 87

# # Evaluación del modelo
# vectores_de_caracteristicas = [model.infer_vector(doc.lower().split()) for doc in lista_opiniones]

# X_train, X_test, y_train, y_test = train_test_split(vectores_de_caracteristicas, lista_atracciones, test_size=0.2, shuffle=True)

# # Entrenar el clasificador KNN
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

# # Predecir las etiquetas para los datos de prueba
# y_pred = knn.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy obtenido con la mejor combinacion de hiperparametros de Doc2Vec: {accuracy}")
# model.save("d2v_01.model")

#-------------------------------------------------------------------K means---------------------------------------------
model= Doc2Vec.load("d2v_01.model")
vectores_de_caracteristicas = [model.infer_vector(doc.lower().split()) for doc in lista_opiniones]
# inertias = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(vectores_de_caracteristicas)
#     inertias.append(kmeans.inertia_)

# # Graficar el método del codo
# plt.plot(range(1, 11), inertias, marker='o')
# plt.title('Método del Codo')
# plt.xlabel('Número de clusters')
# plt.ylabel('Inercia')
# plt.show()

#----------------------------------------------------------------Reduccion de dimensionalidad--------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(vectores_de_caracteristicas)

X_train, X_test, y_train, y_test = train_test_split(X_pca, lista_atracciones, test_size=0.2, random_state=0)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train)

# Guardar el modelo entrenado en un archivo
with open("modelo_kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# with open("modelo_kmeans.pkl", "rb") as f:
#     kmeans = pickle.load(f)

y_train_kmeans = list(kmeans.predict(X_train))

#Visualizar los resultados del clustering con el train
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*', label='Centroides')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Clustering KMeans con PCA con X_train')
plt.legend()
plt.show()

y_test_kmeans = list(kmeans.predict(X_test))
cluster_0 = []
cluster_1 = []
cluster_2 = []

for a,y,t in zip(lista_atracciones,y_test_kmeans,y_test):
    if y == 0:
        cluster_0.append((t,y))
    elif y == 1:
        cluster_1.append((t,y))
    else:
        cluster_2.append((t,y))

print("Opiniones del cluster 0: \n")
for i in range(30):
    print(cluster_0[i])
print("\nOpiniones del cluster 1: \n")
for i in range(30):
    print(cluster_1[i])
print("\nOpiniones del cluster 2: \n")
for i in range(30):
    print(cluster_2[i])

#Visualizar los resultados del clustering con el test
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*', label='Centroides')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Clustering KMeans con PCA con X_test')
plt.legend()
plt.show()

# score = silhouette_score(X_test, kmeans.labels_)
# print("Coeficiente de Silueta:", score)