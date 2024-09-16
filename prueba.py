from ultralytics import YOLO
import cv2
from PIL import Image

# Definir las rutas como variables
ruta_modelo = r'D:\Documentos\Estudios\UNIVERSIDAD\Otros\TFG\Anomaly_Detector\runs\detect\train7\weights\best.pt'  # Reemplaza con la ruta a tu modelo entrenado
ruta_imagen = r'D:\Documentos\Estudios\UNIVERSIDAD\Otros\TFG\Anomaly_Detector\app\backend\uploads\fondo.jpg'  # Reemplaza con la ruta a tu imagen
ruta_imagen_anotada = r'D:\Documentos\Estudios\UNIVERSIDAD\Otros\TFG\Anomaly_Detector\app\backend\results\ej.jpg'  # Reemplaza con la ruta donde deseas guardar la imagen anotada

# Cargar el modelo entrenado
model = YOLO(ruta_modelo)

# Realizar la predicción
resultados = model.predict(ruta_imagen)

# Visualize the results
for i, r in enumerate(resultados):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"./results/results{i}.png")
