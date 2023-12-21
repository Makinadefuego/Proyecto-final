import os
import cv2
import csv
import numpy as np

def extract_save_color_histogram(image_path, output_folder, bins=8):
    # Lee la imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calcula el histograma para cada canal de color
    histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]
    # Normaliza y aplana el histograma para convertirlo en un vector
    histogram = [cv2.normalize(hist, hist).flatten() for hist in histogram]
    histogram_flat = np.concatenate(histogram)

    # Guarda el histograma en un archivo .csv
    with open(os.path.join(output_folder, os.path.basename(image_path) + ".csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(histogram_flat)

if __name__ == "__main__":
    # Lista de imágenes a procesar
    images = os.listdir("objects")

    # Se conservan solo las imágenes .png
    images = [image for image in images if image.endswith(".png")]

    # Crea la carpeta donde se guardarán los histogramas de color
    if not os.path.exists("color_histograms"):
        os.mkdir("color_histograms")

    # Extrae y guarda los histogramas de color de cada imagen
    for image in images:
        extract_save_color_histogram(os.path.join("objects", image), "color_histograms")
