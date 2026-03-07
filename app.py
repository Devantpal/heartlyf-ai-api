from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import os

# Create FastAPI app
app = FastAPI()

# Load the fixed ECG model
model = tf.keras.models.load_model("ecg_model_fixed.keras", compile=False)

# Root route (for testing API)
@app.get("/")
def home():
    return {"message": "Heartlyf ECG AI API Running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: list):

    try:
        # Convert incoming data to numpy array
        arr = np.array(data)

        # Reshape to model input shape
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

# Start server for Render
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )