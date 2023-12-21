#Implementación del algoritmo k-means para la clasificación de datos
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

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
    image_name = os.path.basename(descriptors_path).split('.')[0]  # Obtiene el nombre de la imagen
    with open(descriptors_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            descriptors.append(row + [image_name])  # Añade el nombre de la imagen como etiqueta
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


def select_pca_components(descriptors, variance_threshold=0.95):
    pca = PCA()
    pca.fit(descriptors)
    total_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(total_variance >= variance_threshold) + 1
    return pca, n_components

def apply_pca(descriptors, n_components):
    pca = PCA(n_components=n_components)
    reduced_descriptors = pca.fit_transform(descriptors)
    return reduced_descriptors


    #Se visualizan los clusters en 2D y 3D pero mostrando las componentes principales que más aportan a la varianza

def visualize_clusters(descriptors, labels, n_components = 3):
    componente1 = descriptors[:, 0]
    componente2 = descriptors[:, 1]
    componente3 = descriptors[:, 2]

    if n_components == 2:
        plt.scatter(componente1, componente2, c=labels)
        plt.show()
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(componente1, componente2, componente3, c=labels)
        plt.show()

def visualize_descriptors_with_labels(descriptors, labels):
    fig, ax = plt.subplots()
    for descriptor, label in zip(descriptors, labels):
        ax.scatter(descriptor[0], descriptor[1])  # Asume que los descriptores son 2D
        ax.text(descriptor[0], descriptor[1], label)
    plt.show()


def visualize_descriptors_with_labels_3d(descriptors, labels, samples = 20):
    # descriptors = np.array(descriptors)
    # labels = np.array(labels)
    # random_indices = np.random.choice(len(descriptors), samples, replace=False)
    # descriptors = descriptors[random_indices]
    # labels = labels[random_indices]

    descriptors = np.array(descriptors)
    labels = np.array(labels)

    descriptors = descriptors[:92]
    labels = labels[:92]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for descriptor, label in zip(descriptors, labels):
        ax.scatter(descriptor[0], descriptor[1], descriptor[2])  # Asume que los descriptores son 3D
        ax.text(descriptor[0], descriptor[1], descriptor[2], label)
    
    plt.show()


def mostrar_clusters(descriptors_folder='hu_moments', k=4):
    raw_descriptors = read_all_descriptors(descriptors_folder)
    labels = [d[-1] for d in raw_descriptors]  # Extrae las etiquetas (nombres de las imágenes)
    descriptors = [d[:-1] for d in raw_descriptors]  # Elimina las etiquetas de los descriptores

    descriptors = np.array(descriptors, dtype=np.float32)
    standardized_descriptors = standardize_descriptors(descriptors)

    pca, n_components = select_pca_components(standardized_descriptors)
    pca_descriptors = apply_pca(standardized_descriptors, n_components)

    kmeans = apply_kmeans(pca_descriptors, k)
    visualize_clusters(pca_descriptors, kmeans.labels_)
    visualize_descriptors_with_labels_3d(pca_descriptors, labels)
    elbow_method(pca_descriptors)


if __name__ == "__main__":
    mostrar_clusters()
