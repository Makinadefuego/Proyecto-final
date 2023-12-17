import cv2
import numpy as np

def calcular_flujo_optico(frame_anterior_gray, frame_actual_gray, puntos_anteriores, lk_params):
    # Calcula el flujo óptico
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame_anterior_gray, frame_actual_gray, puntos_anteriores, None, **lk_params)

    # Selecciona los buenos puntos
    buenos_actuales = p1[st == 1]
    buenos_anteriores = puntos_anteriores[st == 1]

    return buenos_actuales, buenos_anteriores

def dibujar_lineas(frame, mask, buenos_actuales, buenos_anteriores):
    # Dibuja las líneas de trayectoria
    for actual, anterior in zip(buenos_actuales, buenos_anteriores):
        a, b = actual.ravel()
        c, d = anterior.ravel()
        a, b, c, d = map(int, [a, b, c, d])
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    return cv2.add(frame, mask)


if __name__ == '__main__':
    # Código principal
    cap = cv2.VideoCapture("Videos/1_low.mp4")
    ret, frame_anterior = cap.read()
    frame_anterior_gray = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)

    # Parámetros para la detección de esquinas de Shi-Tomasi
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    #feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)

    # Parámetros para el flujo óptico de Lucas-Kanade
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    puntos_anteriores = cv2.goodFeaturesToTrack(frame_anterior_gray, mask=None, **feature_params)
    mask = np.zeros_like(frame_anterior)

    while True:
        ret, frame_actual = cap.read()
        if not ret:
            break

        frame_actual_gray = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
        buenos_actuales, buenos_anteriores = calcular_flujo_optico(frame_anterior_gray, frame_actual_gray, puntos_anteriores, lk_params)
        img = dibujar_lineas(frame_actual, mask, buenos_actuales, buenos_anteriores)

        cv2.imshow('Flujo Óptico - Lucas-Kanade', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_anterior_gray = frame_actual_gray.copy()
        puntos_anteriores = buenos_actuales.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()
