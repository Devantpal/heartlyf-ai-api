from fastapi import FastAPI, Form
import numpy as np
import tensorflow as tf

# Create FastAPI app FIRST
app = FastAPI()

print("Loading ECG model...")

# Load TensorFlow model
model = tf.saved_model.load("ecg_saved_model")
infer = model.signatures["serve"]

print("Model loaded successfully")


@app.get("/")
def home():
    return {"status": "ECG AI API running"}


@app.post("/predict")
def predict(ecg_signal: str = Form(...)):

    # Convert string to list
    values = [float(x) for x in ecg_signal.split(",")]

    data = np.array(values, dtype=np.float32)

    # Pad to 720
    if len(data) < 720:
        data = np.pad(data, (0, 720 - len(data)), 'constant')
    else:
        data = data[:720]

    data = data.reshape(1, 720, 1)

    prediction = infer(tf.constant(data))
    result = list(prediction.values())[0].numpy()

    return {"prediction": result.tolist()}