import os
import cv2
import csv
import numpy as np
def extract_save_hu_moments(image_path, output_folder):
    # Lee la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calcula los momentos de la imagen
    moments = cv2.moments(image)

    # Calcula los siete Momentos de Hu
    huMoments = cv2.HuMoments(moments)

    # Logaritmo de los momentos para una mejor visualización
    for i in range(0, 7):
        huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    # Guarda los momentos de Hu en un archivo .csv
    with open(os.path.join(output_folder, os.path.basename(image_path) + ".csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(huMoments.flatten())

if __name__ == "__main__":
    # Lista de imágenes a procesar
    images = os.listdir("objects")

    # Se conservan solo las imágenes .png
    images = [image for image in images if image.endswith(".png")]

    # Crea la carpeta donde se guardarán los momentos de Hu
    if not os.path.exists("hu_moments"):
        os.mkdir("hu_moments")

    # Extrae y guarda los momentos de Hu de cada imagen
    for image in images:
        extract_save_hu_moments(os.path.join("objects", image), "hu_moments")
