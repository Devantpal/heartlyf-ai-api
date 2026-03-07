from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import os

app = FastAPI()

# Rebuild model architecture manually
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(720,1)),
    tf.keras.layers.Conv1D(32,3,activation="relu"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64,3,activation="relu"),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

# Load trained weights
model.load_weights("ecg_model.keras")

@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

@app.post("/predict")
def predict(data:list):

    arr = np.array(data).reshape(1,720,1)

    prediction = model.predict(arr)

    return {"prediction":prediction.tolist()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT",10000))
    uvicorn.run(app,host="0.0.0.0",port=port)