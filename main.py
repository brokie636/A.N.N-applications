import os
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn

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

# Valores mínimos y máximos para desnormalizar
min_sales = -4988.94
max_sales = 693099.36

# modelo de series de tiempo
model = load_model(
    "demand_prediction/model/lstm_model.h5", custom_objects={"mse": MeanSquaredError()}
)

# Simulación de datos de entrenamiento
sequence_length = 30
merged_df = pd.read_csv("./demand_prediction/dataset/merged_df.csv")


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


# Clase para recibir la cantidad de días a predecir
class DaysInput(BaseModel):
    days: int  # Número de días a predecir


@app.post("/predict_day/")
def predict_day(input_data: FeaturesInput):
    # Convertir la entrada a un array numpy con la forma esperada por el modelo
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

    # Realizar predicción
    prediction = model.predict(X_input)[0, 0]

    # Desnormalizar predicción
    prediction = float(prediction * (max_sales - min_sales) + min_sales)

    return {"predicted_sales": prediction}


@app.post("/predict_days/")
def predict_days(input_data: DaysInput):
    """Recibe la cantidad de días y devuelve las predicciones para cada día."""

    # Obtener la última secuencia conocida
    X_last_sequence = np.array(merged_df[features].iloc[-sequence_length:])
    X_last_sequence = X_last_sequence.reshape(1, sequence_length, -1)

    future_predictions = []

    for _ in range(input_data.days):
        next_prediction = model.predict(X_last_sequence)[0, 0]
        future_predictions.append(next_prediction)

        # Desplazar la secuencia e insertar la nueva predicción
        next_input = np.roll(X_last_sequence, shift=-1, axis=1)
        next_input[0, -1, 0] = next_prediction
        X_last_sequence = next_input

    # Desnormalizar predicciones
    future_predictions = (
        np.array(future_predictions) * (max_sales - min_sales) + min_sales
    )

    return {"predicted_sales": future_predictions.tolist()}


# Correr la API en modo local si el script se ejecuta directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
