import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Form

# Create FastAPI app FIRST
app = FastAPI()

print("Loading ECG model...")

# Load TensorFlow SavedModel
model = tf.saved_model.load("ecg_saved_model")
infer = model.signatures["serve"]

print("Model loaded successfully")


@app.get("/")
def home():
    return {"status": "ECG AI API running"}


@app.post("/predict")
def predict(ecg_signal: list = Form(...)):

    data = np.array(ecg_signal, dtype=np.float32)

    # Pad or trim to 720 samples
    if len(data) < 720:
        data = np.pad(data, (0, 720 - len(data)), 'constant')
    else:
        data = data[:720]

    data = data.reshape(1, 720, 1)

    prediction = infer(tf.constant(data))

    result = list(prediction.values())[0].numpy()

    return {"prediction": result.tolist()}