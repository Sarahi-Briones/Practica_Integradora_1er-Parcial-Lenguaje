from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de Iris (o utilizar tus propios datos)
data = load_iris()
X = data.data  # Características

# Reducción de dimensionalidad con PCA a 2 componentes principales
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Aplicar KMeans en los datos transformados por PCA
kmeans = KMeans(n_clusters=3)  # Especificar el número de clusters
kmeans.fit(X_pca)
y_kmeans = kmeans.predict(X_pca)

print(y_kmeans)