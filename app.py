@app.post("/predict")
def predict(ecg_signal: list):

    data = np.array(ecg_signal, dtype=np.float32)

    # pad or trim to 720 samples
    if len(data) < 720:
        data = np.pad(data, (0, 720 - len(data)), 'constant')
    else:
        data = data[:720]

    data = data.reshape(1, 720, 1)

    prediction = infer(tf.constant(data))

    result = prediction["output_0"].numpy()

    return {"prediction": result.tolist()}