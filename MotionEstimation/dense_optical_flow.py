import cv2
import numpy as np
from recortarObjetosEscena import recortar_y_guardar_objetos



def calcular_flujo_optico_farneback(prvs, next, farneback_params):
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, **farneback_params)
    magn, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magn, ang

def visualizar_flujo_optico(magn, ang):
    magn_norm = cv2.normalize(magn, None, 0, 255, cv2.NORM_MINMAX)
    ang_scaled = ang * 180 / np.pi / 2

    hsv = np.zeros((magn.shape[0], magn.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang_scaled
    hsv[..., 1] = 255
    hsv[..., 2] = magn_norm
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def segmentar_movimiento(magn, umbral):
    mask = magn > umbral
    segmentacion = np.zeros_like(magn, dtype=np.uint8)
    segmentacion[mask] = 255

    kernel = np.ones((5, 5), np.uint8)
    segmentacion = cv2.morphologyEx(segmentacion, cv2.MORPH_ERODE, kernel)
    segmentacion = cv2.morphologyEx(segmentacion, cv2.MORPH_DILATE, kernel)

    return segmentacion

#Función para usar la mascara de segmentación
def segmentacion(frame, segmentacion):
    frame = cv2.bitwise_and(frame, frame, mask=segmentacion)
    return frame

if __name__ == '__main__':
    # Código principal
    cap = cv2.VideoCapture("Videos/1.mp4")
    farneback_params = dict(pyr_scale=0.5, levels=10, winsize=40, iterations=3, poly_n=3, poly_sigma=1.2, flags=0)
    ret, frame = cap.read()
    prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    umbral_magnitud = 2  # Ajustar según sea necesario
    i = 0
    j = 0
    while True:
        i += 1
        j += 1
        ret, frame = cap.read()
        if not ret:
            break

        if i%2 == 0:
            continue

      
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        magn, ang = calcular_flujo_optico_farneback(prvs, next, farneback_params)
        
        imagen_flujo = visualizar_flujo_optico(magn, ang)
        mascara_segmentacion = segmentar_movimiento(magn, umbral_magnitud)
        imagen_segmentacion = segmentacion(frame, mascara_segmentacion)
        
        if j == 1:
            recortar_y_guardar_objetos(frame, mascara_segmentacion, output_folder="objects")

        cv2.imshow('Flujo Óptico Denso', imagen_flujo)
        cv2.imshow('Máscara de Segmentación', mascara_segmentacion)
        cv2.imshow('Segmentación de Movimiento', imagen_segmentacion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prvs = next

    cap.release()
    cv2.destroyAllWindows()
