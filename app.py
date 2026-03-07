import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI

app = FastAPI()

print("Loading ECG model...")

model = tf.keras.models.load_model(
    "ecg_model_clean.keras",
    compile=False
)

print("Model loaded successfully")

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(ecg_signal: list):

    try:
        data = np.array(ecg_signal)
        data = data.reshape(1,720,1)

        prediction = model.predict(data)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}