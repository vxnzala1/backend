from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import shutil
import os

app = FastAPI()

# Configuración de CORS para permitir solicitudes desde tu frontend (enlace de Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cerambix.go-capricornio.com"],  # Cambia esto por la URL de tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir las rutas como variables (rutas relativas para Render)
ruta_modelo = r'./best.pt'  # Ruta relativa al modelo entrenado
ruta_carpeta_imagen = r'./uploads'  # Carpeta para guardar las imágenes subidas (ruta relativa)
ruta_carpeta_imagen_anotada = r'./results'  # Carpeta para guardar las imágenes procesadas (ruta relativa)

# Asegurarse de que las carpetas existen (si no, las crea)
os.makedirs(ruta_carpeta_imagen, exist_ok=True)
os.makedirs(ruta_carpeta_imagen_anotada, exist_ok=True)

# Montar la carpeta 'results' como archivos estáticos
app.mount("/results", StaticFiles(directory=ruta_carpeta_imagen_anotada), name="results")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Guardar la imagen subida
        file_location = os.path.join(ruta_carpeta_imagen, file.filename)
        with open(file_location, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There was an error uploading the file: {str(e)}")
    finally:
        file.file.close()

    # Ruta de la imagen anotada
    ruta_imagen_anotada = os.path.join(ruta_carpeta_imagen_anotada, file.filename)
    ruta_imagen = file_location

    # Cargar el modelo YOLO
    model = YOLO(ruta_modelo)

    # Realizar la predicción y guardar la imagen anotada
    resultados = model.predict(ruta_imagen)
    anotada_img = resultados[0].plot()  # Genera una imagen con las cajas de las predicciones
    imagen_anotada_pil = Image.fromarray(anotada_img)

    # Guardar la imagen anotada en la carpeta "results"
    imagen_anotada_pil.save(ruta_imagen_anotada)

    # Devolver la URL de la imagen procesada al frontend
    processed_image_url = f"https://backend-1-gaka.onrender.com/results/{file.filename}"
    return JSONResponse({"processed_image_url": processed_image_url})
