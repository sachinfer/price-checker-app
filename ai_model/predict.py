# ai_model/predict.py
import numpy as np
from tensorflow.keras.models import load_model
from .preprocess import preprocess_image  # your preprocessing logic

model = load_model('ai_model/trained_model.h5')  # ✅ relative path

def predict_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed, axis=0))
    return prediction.argmax(axis=1)[0]  # e.g., return predicted class index
