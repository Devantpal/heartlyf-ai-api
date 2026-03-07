from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

app = FastAPI()

# Fix compatibility with old model
class FixedInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs["input_shape"] = batch_shape[1:]
        super().__init__(**kwargs)

model = load_model(
    "ecg_model.h5",
    compile=False,
    custom_objects={"InputLayer": FixedInputLayer}
)

@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

@app.post("/predict")
def predict(data: list):

    arr = np.array(data)
    arr = arr.reshape(1, 720, 1)

    prediction = model.predict(arr)

    return {"prediction": prediction.tolist()}