import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
# Cargamos el video
cap = cv2.VideoCapture("Videos/1.mp4")

# Parámetros para la detección de esquinas de Shi-Tomasi
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parámetros para el flujo óptico de Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Tomamos el primer frame y encontramos esquinas en él
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Creamos una máscara para dibujar el flujo óptico
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculamos el flujo óptico
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Seleccionamos los buenos puntos
    good_new = p1[st == 1]
    good_old = p0[st == 1]

        # Dibujamos las trazas
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # Convertimos los puntos a enteros
        a, b, c, d = map(int, [a, b, c, d])

        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    
    img = cv2.add(frame, mask)

    cv2.imshow('Flujo Óptico - Lucas-Kanade', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    
    # Actualizamos el frame anterior y los puntos anteriores
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
