from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image
import shutil
import os
import io

app = FastAPI()

# Configuración de CORS para permitir solicitudes desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por el origen específico de tu frontend en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir las rutas como variables
ruta_modelo = r'./best.pt'  # Reemplaza con la ruta a tu modelo entrenado
ruta_carpeta_imagen = r'./uploads'  # Usa una ruta relativa para mantener la portabilidad
ruta_carpeta_imagen_anotada = r'./results'  # Usa una ruta relativa para mantener la portabilidad

# Asegurarse de que las carpetas existen
os.makedirs(ruta_carpeta_imagen, exist_ok=True)
os.makedirs(ruta_carpeta_imagen_anotada, exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Guardar la imagen cargada en la carpeta de uploads
        file_location = os.path.join(ruta_carpeta_imagen, file.filename)
        with open(file_location, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There was an error uploading the file: {str(e)}")
    finally:
        file.file.close()

    ruta_imagen_anotada = os.path.join(ruta_carpeta_imagen_anotada, file.filename)
    ruta_imagen = file_location

    # Cargar el modelo entrenado
    model = YOLO(ruta_modelo)

    # Realizar la predicción
    resultados = model.predict(ruta_imagen)

    # Guardar la imagen anotada temporalmente para devolverla al frontend
    anotada_img = resultados[0].plot()  # Esto genera una imagen con las cajas de las predicciones
    imagen_anotada_pil = Image.fromarray(anotada_img)

    # Convertir la imagen anotada en un buffer de memoria
    buffer = io.BytesIO()
    imagen_anotada_pil.save(buffer, format="JPEG")
    buffer.seek(0)

    # Devolver la imagen procesada como una respuesta de streaming
    return StreamingResponse(buffer, media_type="image/jpeg")
