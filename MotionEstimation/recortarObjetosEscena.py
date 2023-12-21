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
    j = len(os.listdir(output_folder))
    print(j)
    for i, cnt in enumerate(contours):
        #j es el número del ultimo objeto guardado
        # Obtiene el rectángulo delimitador para cada contorno
        x, y, w, h = cv2.boundingRect(cnt)

        #Si el tamaño del objeto es muy pequeño, se ignora
        if w < 20 and h < 20:
            continue

        # Recorta la región del objeto del frame original
        objeto_recortado = frame[y:y+h, x:x+w]

        # Encuentra un nombre de archivo no utilizado
        nombre_archivo = ""
        numero_archivo = 0
        while True:
            nombre_archivo = os.path.join(output_folder, f"objeto_{numero_archivo}.png")
            if not os.path.exists(nombre_archivo):
                break
            numero_archivo += 1

        # Guarda el objeto recortado con el nombre de archivo único
        cv2.imwrite(nombre_archivo, objeto_recortado)


    print(f"Se han guardado {len(contours)} objetos en la carpeta '{output_folder}'.")