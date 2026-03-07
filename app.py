from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("ecg_model.h5", compile=False)

@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

@app.post("/predict")
def predict(data: list):

    arr = np.array(data).reshape(1,720,1)

    prediction = model.predict(arr)

    result = int(np.argmax(prediction))

    confidence = float(np.max(prediction))

    return {
        "prediction": result,
        "confidence": confidence
    }