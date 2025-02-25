import os
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.losses import MeanSquaredError

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS para permitir todas las solicitudes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# --- Cargar Modelo de Predicción de Series de Tiempo ---
min_sales = -4988.94
max_sales = 693099.36
sequence_length = 30
model = load_model(
    "demand_prediction/model/lstm_model.h5", custom_objects={"mse": MeanSquaredError()}
)
merged_df = pd.read_csv("./demand_prediction/dataset/merged_df.csv")

features = [
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Size",
    "IsHoliday",
]


class FeaturesInput(BaseModel):
    Temperature: float
    Fuel_Price: float
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float
    CPI: float
    Unemployment: float
    Size: int
    IsHoliday: int


class DaysInput(BaseModel):
    days: int  # Número de días a predecir


@app.post("/predict_day/")
def predict_day(input_data: FeaturesInput):
    X_input = np.array(
        [
            [
                input_data.Temperature,
                input_data.Fuel_Price,
                input_data.MarkDown1,
                input_data.MarkDown2,
                input_data.MarkDown3,
                input_data.MarkDown4,
                input_data.MarkDown5,
                input_data.CPI,
                input_data.Unemployment,
                input_data.Size,
                input_data.IsHoliday,
            ]
        ]
    ).reshape(1, 1, -1)

    prediction = model.predict(X_input)[0, 0]
    prediction = float(prediction * (max_sales - min_sales) + min_sales)
    return {"predicted_sales": prediction}


@app.post("/predict_days/")
def predict_days(input_data: DaysInput):
    X_last_sequence = np.array(merged_df[features].iloc[-sequence_length:]).reshape(
        1, sequence_length, -1
    )
    future_predictions = []

    for _ in range(input_data.days):
        next_prediction = model.predict(X_last_sequence)[0, 0]
        future_predictions.append(next_prediction)
        X_last_sequence = np.roll(X_last_sequence, shift=-1, axis=1)
        X_last_sequence[0, -1, 0] = next_prediction

    future_predictions = (
        np.array(future_predictions) * (max_sales - min_sales) + min_sales
    )
    return {"predicted_sales": future_predictions.tolist()}


# --- Cargar Modelo de Predicción de Imágenes ---
image_model = load_model("product_image_prediction/model/modelo_entrenado.h5")
print(image_model.input_shape)


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)  # Redimensionar la imagen
    img_array = image.img_to_array(img) / 255.0  # Normalizar al rango [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión del batch
    return img_array


@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        img_path = os.path.join(temp_dir, file.filename)

        with open(img_path, "wb") as f:
            f.write(await file.read())

        classes = ["Jeans", "Sofa", "T-shirt", "TV"]
        img_array = load_and_preprocess_image(img_path, target_size=(224, 224))
        prediction = image_model.predict(img_array)

        # Aquí puedes cambiar la lógica de salida según tu modelo
        predicted_class_index = np.argmax(prediction, axis=1).tolist()[0]
        predicted_class = classes[predicted_class_index]
        print(predicted_class)
        os.remove(img_path)  # Eliminar imagen después de la predicción
        return {"predicted_class": predicted_class}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error procesando la imagen: {str(e)}"
        )


# --- Correr la API ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
