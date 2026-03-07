import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print("Loading ECG model...")
        model = tf.keras.models.load_model("ecg_model_clean.keras", compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print("MODEL LOAD ERROR:", e)


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/predict")
def predict(ecg_signal: list):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        data = np.array(ecg_signal)
        data = data.reshape(1,720,1)

        prediction = model.predict(data)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}