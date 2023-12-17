import cv2
import numpy as np
import os

def recortar_y_guardar_objetos(frame, mask, output_folder="objects"):
    """
    Esta función toma un frame y una máscara binaria, recorta los objetos identificados
    en la máscara y los guarda como imágenes PNG en una carpeta especificada.

    :param frame: Imagen de entrada.
    :param mask: Máscara binaria con los objetos de interés.
    :param output_folder: Carpeta donde se guardarán los objetos recortados.
    """

    # Asegura que la máscara esté en el formato correcto
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    #Al frame se le aplica la mascara para quedar solo con los objetos
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    

    # Crea la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Encuentra los contornos de los objetos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        # Obtiene el rectángulo delimitador para cada contorno
        x, y, w, h = cv2.boundingRect(cnt)

        # Recorta la región del objeto del frame original
        objeto_recortado = frame[y:y+h, x:x+w]

        # Guarda el objeto recortado como imagen PNG
        cv2.imwrite(os.path.join(output_folder, f"objeto_{i}.png"), objeto_recortado)

    print(f"Se han guardado {len(contours)} objetos en la carpeta '{output_folder}'.")