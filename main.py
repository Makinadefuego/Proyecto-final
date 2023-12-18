from MotionEstimation.dense_optical_flow import *
from MotionEstimation.sparse_optical_flow import *
from MotionEstimation.raft_implentation import *
from MotionEstimation.recortarObjetosEscena import recortar_y_guardar_objetos
from FeatureExtraction.apply_sift import *
from Classification.k_means import *


#Se Aplica el algoritmo de RAFT para obtener el flujo optico según el archivo raft_implentation.py
def process_video(video_path, model_path, device='cpu'):
    # Carga el modelo RAFT
    model = load_model(model_path)
    model.to(device)
    model.eval()

    # Abre el video
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    
    if not ret:
        print("No se pudo leer el video.")
        return

    i = 0
    j = 0
    while True:
        i += 1
        j += 1
        if i % 2 == 0:
            continue
        
        ret, frame2 = cap.read()
        if not ret:
            break

        # Calcula el flujo óptico y las visualizaciones
        flow_low, flow_up = inference(model, frame1, frame2, device, test_mode=True)
        flow_up_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        
        flow_viz = visualize_optical_flow(flow_up_np)
        filtered_flow = filter_optical_flow(flow_up_np)
        mascara = segment_movement(flow_up_np)
        segmented_movement = segmentacion(frame2, mascara)

        
        recortar_y_guardar_objetos(frame2, mascara, output_folder="objects")

        if j == 9:
            break
        # Muestra los resultados
        cv2.imshow("Optical Flow", flow_viz)
        cv2.imshow("Filtered Optical Flow", visualize_optical_flow(filtered_flow))
        cv2.imshow("Segmented Movement", segmented_movement)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame1 = frame2

    cap.release()
    cv2.destroyAllWindows()

# Uso de la función
video_path = "Videos/4.mp4"
model_path = "MotionEstimation/RAFT/models/raft-things.pth"
process_video(video_path, model_path)


#Se extran las caracteristicas

    # Lista de imágenes a procesar
images = os.listdir("objects")

    #Se conservan solo las imágenes .jpg
images = [image for image in images if image.endswith(".png")]
print(images)

    # Crea la carpeta donde se guardarán los descriptores
if not os.path.exists("sift_features"):
    os.mkdir("sift_features")

for image in os.listdir("objects"):
    try:
        # Intenta extraer y guardar las características SIFT
        extract_save_sift_features(os.path.join("objects", image), "sift_features")
    except RuntimeError as e:
        # Si se produce un RuntimeError, verifica si es el error específico
        if "SIFT found no features" in str(e):
            print(f"No se encontraron características en {image}, omitiendo...")
            # Aquí puedes agregar cualquier otra lógica que necesites
        else:
            # Si es un RuntimeError diferente, puedes optar por volver a lanzarlo
            raise

mostrar_clusters()

