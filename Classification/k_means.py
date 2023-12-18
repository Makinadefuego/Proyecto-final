#Implementación del algoritmo k-means para la clasificación de datos
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Metodo del codo para determinar el numero de clusters
def elbow_method(descriptors):
    inertias = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(descriptors)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, 10), inertias)
    plt.show()
    

#Función para leer los descriptores de las imágenes
def read_descriptors(descriptors_path):
    descriptors = []
    with open(descriptors_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            descriptors.append(row)
    return descriptors


#Función para leer los descriptores de todas las imágenes
def read_all_descriptors(descriptors_folder):
    descriptors = []
    for file in os.listdir(descriptors_folder):
        descriptors += read_descriptors(os.path.join(descriptors_folder, file))
    return descriptors


#Función para aplicar el algoritmo k-means  
def apply_kmeans(descriptors, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(descriptors)
    return kmeans

# Función para estandarizar los descriptores (opcional pero recomendado)
def standardize_descriptors(descriptors):
    scaler = StandardScaler()
    scaled_descriptors = scaler.fit_transform(descriptors)
    return scaled_descriptors

# Función para aplicar PCA y reducir a 2 dimensiones
def apply_pca(descriptors, n_components=3):
    pca = PCA(n_components=n_components)
    reduced_descriptors = pca.fit_transform(descriptors)
    return reduced_descriptors

# Función para visualizar los resultados de k-means con PCA
def visualize_clusters(pca_descriptors, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_descriptors[:, 0], pca_descriptors[:, 1], pca_descriptors[:, 2], c=labels)
    plt.show()

# Función principal
def mostrar_clusters():
    descriptors_folder = 'sift_features'
    k = 4  # Número de clusters en k-means

    # Leer y preparar los descriptores
    descriptors = read_all_descriptors(descriptors_folder)
    descriptors = np.array(descriptors, dtype=np.float32)  # Asegúrate de que sean floats

    # Estandarizar y aplicar PCA
    standardized_descriptors = standardize_descriptors(descriptors)
    pca_descriptors = apply_pca(standardized_descriptors)

    # Aplicar k-means
    kmeans = apply_kmeans(pca_descriptors, k)

    # Visualizar los resultados
    visualize_clusters(pca_descriptors, kmeans.labels_)
    elbow_method(pca_descriptors)

if __name__ == "__main__":
    mostrar_clusters()