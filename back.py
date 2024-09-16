from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

import shutil
import os

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

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
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

    # Guardar la imagen anotada
    resultados[0].save(filename=ruta_imagen_anotada)

    # Devolver la URL de la imagen procesada al frontend
    return JSONResponse({"message": "Successfully uploaded", "processed_image_url": ruta_imagen_anotada})
