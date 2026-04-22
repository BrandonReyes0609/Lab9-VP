import cv2
import time
from ultralytics import YOLO

# YOLOv8n (nano): versión ligera, preentrenada en COCO dataset es decir con objetos cotidianos.
modelo = YOLO("yolov8n.pt")  

UMBRAL_CONFIANZA = 0.25
UMBRAL_IOU = 0.45

captura = cv2.VideoCapture("video.MOV")

if not captura.isOpened():
    exit()

ancho_original = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_original = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

ALTO_MAXIMO = 720
proporcion = ancho_original / alto_original

if alto_original > ALTO_MAXIMO:
    alto_display = ALTO_MAXIMO
    ancho_display = int(alto_display * proporcion)
else:
    alto_display = alto_original
    ancho_display = ancho_original

print(f"Resolución: {ancho_display}x{alto_display}")

while True:
    ret, cuadro = captura.read()
    if not ret:
        break

    cuadro = cv2.resize(cuadro, (ancho_display, alto_display))

    tiempo_inicio = time.time()

    resultados = modelo(
        cuadro,
        conf=UMBRAL_CONFIANZA,
        iou=UMBRAL_IOU,
        verbose=False
    )
    tiempo_fin = time.time()
    tiempo_transcurrido = tiempo_fin - tiempo_inicio
    fps = 1.0 / tiempo_transcurrido if tiempo_transcurrido > 0 else 0.0

    detecciones = resultados[0].boxes

    escala_texto = max(0.5, min(ancho_display, alto_display) / 800)
    grosor_texto = max(1, int(escala_texto * 2))
    grosor_caja = max(1, int(escala_texto * 2.5))

    for i in range(len(detecciones)):
        x1, y1, x2, y2 = detecciones.xyxy[i].cpu().numpy().astype(int)

        confianza = float(detecciones.conf[i].cpu().numpy())

        id_clase = int(detecciones.cls[i].cpu().numpy())
        nombre_clase = modelo.names[id_clase]

        etiqueta = f"{nombre_clase} {confianza:.2f}"

        cv2.rectangle(cuadro, (x1, y1), (x2, y2), (0, 255, 0), grosor_caja)

        (ancho_texto, alto_texto), linea_base = cv2.getTextSize(
            etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala_texto, grosor_texto
        )

        cv2.rectangle(
            cuadro,
            (x1, y1 - alto_texto - linea_base - 5),
            (x1 + ancho_texto, y1),
            (0, 255, 0),
            -1,  
        )

        cv2.putText(
            cuadro,
            etiqueta,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            escala_texto,
            (0, 0, 0),  
            grosor_texto,
        )

    texto_fps = f"FPS: {fps:.1f}"
    cv2.putText(
        cuadro,
        texto_fps,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),  
        2,
    )

    cv2.imshow("Task 3", cuadro)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
