from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = FastAPI()

# Load ECG AI model
model = load_model("ecg_model.h5", compile=False)

@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

@app.post("/predict")
def predict(data: list):

    arr = np.array(data)
    arr = arr.reshape(1, 720, 1)

    prediction = model.predict(arr)

    return {
        "prediction": prediction.tolist()
    }