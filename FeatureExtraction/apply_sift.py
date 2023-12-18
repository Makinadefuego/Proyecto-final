import os
import cv2
import random
from skimage import feature 
import shutil
from skimage import data
from skimage import transform
from skimage.feature import SIFT
from skimage.color import rgb2gray
import csv

def extract_save_sift_features(image_path, output_folder):
    # Lee la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Inicializa el detector sift
    sift = feature.SIFT()
    
    # Detecta los puntos clave y los descriptores de la imagen
    sift.detect_and_extract(image)

    # Guarda los descriptores en un archivo .csv
    with open(os.path.join(output_folder, os.path.basename(image_path) + ".csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(sift.descriptors)

if __name__ == "__main__":
    # Lista de imágenes a procesar
    images = os.listdir("objects")

    #Se conservan solo las imágenes .jpg
    images = [image for image in images if image.endswith(".png")]
    print(images)

    # Crea la carpeta donde se guardarán los descriptores
    if not os.path.exists("sift_features"):
        os.mkdir("sift_features")

    # Extrae y guarda los descriptores de cada imagen
    for image in images:
        extract_save_sift_features(os.path.join("objects", image), "sift_features")
