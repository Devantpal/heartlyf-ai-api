import numpy as np
import tensorflow as tf
from fastapi import FastAPI

app = FastAPI()

print("Loading ECG model...")

# Load TensorFlow SavedModel
model = tf.saved_model.load("ecg_saved_model")

infer = model.signatures["serving_default"]

print("Model loaded successfully")


@app.get("/")
def home():
    return {"status": "ECG AI API running"}


@app.post("/predict")
def predict(ecg_signal: list):

    data = np.array(ecg_signal).reshape(1,720,1).astype("float32")

    prediction = infer(tf.constant(data))

    return {"prediction": str(prediction)}