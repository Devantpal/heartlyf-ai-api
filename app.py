from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Load model using legacy loader (fixes batch_shape issue)
model = tf.keras.models.load_model(
    "ecg_model.keras",
    compile=False,
    safe_mode=False
)

@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

@app.post("/predict")
def predict(data: list):

    arr = np.array(data).reshape(1, 720, 1)
    prediction = model.predict(arr)

    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)