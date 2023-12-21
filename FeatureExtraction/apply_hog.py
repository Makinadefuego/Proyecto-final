import os
import cv2
import csv
from skimage.feature import hog

def extract_save_hog_features(image_path, output_folder, size=(128, 128)):
    # Lee la imagen y la redimensiona
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)

    # Calcula las características HOG de la imagen redimensionada
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, block_norm='L2-Hys')
    
    # Guarda las características HOG en un archivo .csv
    with open(os.path.join(output_folder, os.path.basename(image_path) + ".csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(hog_features)

if __name__ == "__main__":
    # Lista de imágenes a procesar
    images = os.listdir("objects")

    # Se conservan solo las imágenes .png
    images = [image for image in images if image.endswith(".png")]

    # Crea la carpeta donde se guardarán las características HOG
    if not os.path.exists("hog_features"):
        os.mkdir("hog_features")

    # Extrae y guarda las características HOG de cada imagen
    for image in images:
        extract_save_hog_features(os.path.join("objects", image), "hog_features")
