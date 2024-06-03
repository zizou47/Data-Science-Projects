# Function to make prediction using the loaded model
def predict(model, data):
    prediction = model.predict([[data.diagonal, data.height_right, data.margin_low, data.margin_up, data.length]])
    return prediction[0]
