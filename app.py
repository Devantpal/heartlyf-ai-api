from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the ECG AI model (new .keras format)
model = tf.keras.models.load_model("ecg_model.keras")

@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

@app.post("/predict")
def predict(data: list):

    try:
        # Convert incoming data to numpy array
        arr = np.array(data)

        # Reshape according to model input (720 ECG samples)
        arr = arr.reshape(1, 720, 1)

        # Run prediction
        prediction = model.predict(arr)

        return {
            "prediction": prediction.tolist()
        }

    except Exception as e:
        return {
            "error": str(e)
        }