@app.post("/predict")
def predict(ecg_signal: str = Form(...)):

    # Convert string to list
    values = [float(x) for x in ecg_signal.split(",")]

    data = np.array(values, dtype=np.float32)

    # Pad or trim to 720
    if len(data) < 720:
        data = np.pad(data, (0, 720 - len(data)), 'constant')
    else:
        data = data[:720]

    data = data.reshape(1, 720, 1)

    prediction = infer(tf.constant(data))

    result = list(prediction.values())[0].numpy()

    return {"prediction": result.tolist()}