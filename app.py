import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
import uvicorn

# Create FastAPI app
app = FastAPI(title="Heartlyf ECG AI API")

# Load ECG model
print("Loading ECG model...")

model = tf.keras.models.load_model(
    "ecg_model_clean.keras",
    compile=False
)

print("Model loaded successfully")

# -----------------------------
# Root route (API test)
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Heartlyf ECG AI API"
    }


# -----------------------------
# Prediction route
# -----------------------------
@app.post("/predict")
def predict(ecg_signal: list):

    try:
        # Convert input to numpy array
        data = np.array(ecg_signal)

        # Reshape for model input
        data = data.reshape(1, 720, 1)

        # Run prediction
        prediction = model.predict(data)

        result = prediction.tolist()

        return {
            "prediction": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )